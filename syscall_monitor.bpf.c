//go:build ignore

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "page_fault.h"

/*
 * memcpy inside a BPF program is not a libbpf helper; it resolves to a
 * normal C library call, which clang refuses to emit against the BPF
 * target. Historically libbpf headers (and older clang versions) were
 * lenient and allowed the implicit declaration to slip through, but
 * clang 14+ with the BPF target rejects the implicit prototype:
 *
 *   syscall_monitor.bpf.c:159:5: error: call to undeclared library
 *   function 'memcpy' with type 'void *(void *, const void *, unsigned
 *   long)'; ISO C99 and later do not support implicit function
 *   declarations
 *
 * This fails the build on Debian trixie (clang 19, kernel 6.16) even
 * though the code was compiling fine on older toolchains.
 *
 * The compiler provides __builtin_memcpy, which clang happily lowers to
 * the BPF memory moves the verifier already understands. Route every
 * memcpy() in this translation unit through the builtin via a macro
 * shim so the BPF source stays readable and future copies do not have
 * to remember the quirk.
 */
#ifndef memcpy
#define memcpy(dst, src, n) __builtin_memcpy((dst), (src), (n))
#endif

/*
 * PT_REGS_PARM6 is missing in many libbpf versions because
 * the 6th syscall arg doesn't pass through a standard calling
 * convention register on all archs. Define it per-arch if absent.
 */
#ifndef PT_REGS_PARM6
  #if defined(__TARGET_ARCH_x86)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->r9)
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), r9)
  #elif defined(__TARGET_ARCH_arm64)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->regs[5])
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), regs[5])
  #elif defined(__TARGET_ARCH_s390)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->gprs[7])
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), gprs[7])
  #elif defined(__TARGET_ARCH_powerpc)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->gpr[8])
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), gpr[8])
  #elif defined(__TARGET_ARCH_riscv)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->a5)
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), a5)
  #elif defined(__TARGET_ARCH_loongarch)
    #define PT_REGS_PARM6(x)       ((__u64)(x)->regs[9])
    #define PT_REGS_PARM6_CORE(x)  BPF_CORE_READ((x), regs[9])
  #else
    #error "PT_REGS_PARM6: unsupported architecture"
  #endif
#endif

#define MAX_COMM_LEN 16
#define MAX_ENTRIES 8192
#define MAX_MMAP_REGIONS 1024
#define PAGE_SIZE  4096

/* file access modes */
#define O_ACCMODE   00000003
#define O_RDONLY    00000000
#define O_WRONLY    00000001
#define O_RDWR      00000002

/* file creation and status flags */
#define O_CREAT     00000100
#define O_EXCL      00000200
#define O_NOCTTY    00000400
#define O_TRUNC     00001000
#define O_APPEND    00002000
#define O_NONBLOCK  00004000
#define O_DSYNC     00010000
#define FASYNC      00020000
#define O_DIRECT    00040000
#define O_LARGEFILE 00100000
#define O_DIRECTORY 00200000
#define O_NOFOLLOW  00400000
#define O_CLOEXEC   02000000
#define O_SYNC      04000000
#define O_PATH      010000000

enum io_direction {
	READ,
	WRITE,
	VREAD,
	VWRITE
};

// Syscall event structure
struct syscall_event {
    u64 timestamp;
    u32 pid;
    u32 syscall_nr;
    u32 fd;
    u64 size;
    s64 offset;
    char comm[MAX_COMM_LEN];
    char filename[256];
    u32 open_flags_hex;
    char open_flags_str[128];
    enum io_direction ddir;
    long ret;  /* holds the number of bytes transferred */
    long error_code;  /* holds the error code returned by the syscall */
};

// Maps for syscall statistics
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u64);
} syscall_sizes SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u64);
} syscall_counts SEC(".maps");

// Ring buffer for detailed events
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 8 * 1024 * 1024);  // 8MB ring buffer (increased from 256KB)
} events SEC(".maps");

// struct used to map the timestamps to pid_tgid
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u64);     // pid_tgid
    __type(value, u64);   // timestamp captured at mmap entry
} mmap_entry_map SEC(".maps");


// Global flag to control detailed logging
volatile const bool detailed_logging = true;
volatile const unsigned int sampling_rate = 1;  // Sample 1 out of every N events (1 = capture all)
/* __u32 (not u32): u32 is a vmlinux.h-only typedef, so bpftool would emit an
 * undeclared "u32" field in the userspace skeleton struct otherwise. */
volatile const __u32 target_tgid = 0;  // 0 = trace every process, otherwise only this tgid

static __always_inline bool is_target_pid(u32 tgid)
{
    return target_tgid == 0 || tgid == target_tgid;
}

// enter/exit join: stash read/write-family enter args by TID so sys_exit can attach the bytes the
// syscall RETURNED. Without this, enter events carry (fd,count,offset) but ret=-1, and the generic
// sys_exit carries ret but fd=-1/count=1/offset=0 -- the two are never correlated, so neither
// requested-vs-returned nor per-fd byte accounting is recoverable. This restores a single joined
// record {fd, offset, requested count, returned bytes} emitted at completion.
struct io_ctx {
    u32 syscall_nr;
    u32 fd;
    u64 count;   // requested bytes (or iovcnt for readv/writev)
    u64 offset;  // absolute offset for pread/pwrite, 0 otherwise
};
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u64);            // bpf_get_current_pid_tgid()
    __type(value, struct io_ctx);
} io_inflight SEC(".maps");

static __always_inline void stash_io_enter(u32 syscall_nr, u32 fd, u64 count, u64 offset)
{
    u64 tid = bpf_get_current_pid_tgid();
    struct io_ctx c = {};
    c.syscall_nr = syscall_nr;
    c.fd = fd;
    c.count = count;
    c.offset = offset;
    bpf_map_update_elem(&io_inflight, &tid, &c, BPF_ANY);
}

static __always_inline void update_stats(u32 syscall_nr, u64 size)
{
    u64 *val;
    u64 init_val = size;
    u64 count = 1;

    val = bpf_map_lookup_elem(&syscall_sizes, &syscall_nr);
    if (val) {
        __sync_fetch_and_add(val, size);
    } else {
        bpf_map_update_elem(&syscall_sizes, &syscall_nr, &init_val, BPF_ANY);
    }

    val = bpf_map_lookup_elem(&syscall_counts, &syscall_nr);
    if (val) {
        __sync_fetch_and_add(val, 1);
    } else {
        bpf_map_update_elem(&syscall_counts, &syscall_nr, &count, BPF_ANY);
    }
}

static __always_inline void log_event_with_owner(u32 syscall_nr, u32 fd, u64 size, s64 offset, char filename[256], int open_flags_hex, char open_flags_str[128], long ret, long error_code, const struct fault_io_owner *owner)
{
    struct syscall_event *event;

    if (!detailed_logging)
        return;

    // Sampling: only log every Nth event
    if (sampling_rate > 1) {
        static __u32 counter = 0;
        counter++;
        if (counter % sampling_rate != 0)
            return;
    }

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return;

    event->timestamp = bpf_ktime_get_ns();
    event->pid = owner ? owner->pid_tgid >> 32 : bpf_get_current_pid_tgid() >> 32;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    if (syscall_nr == 9) {   /* mmap */
        if (filename[0] == '\0') {
            /*
             * mmap ENTRY — capture fresh timestamp, save it
             * in map keyed by pid_tgid for exit to reuse
             */
            u64 ts = bpf_ktime_get_ns();
            event->timestamp = ts;
            bpf_map_update_elem(&mmap_entry_map, &pid_tgid, &ts, BPF_ANY);

        } else {
            /*
             * mmap EXIT — reuse the exact timestamp from entry
             * so both log records are correlated by (pid_tgid, timestamp)
             */
            u64 *saved_ts = bpf_map_lookup_elem(&mmap_entry_map, &pid_tgid);
            event->timestamp = saved_ts ? *saved_ts : bpf_ktime_get_ns();
            bpf_map_delete_elem(&mmap_entry_map, &pid_tgid);  // cleanup
        }
    } else {
        // all other syscalls — fresh timestamp
        event->timestamp = bpf_ktime_get_ns();
    }
    event->syscall_nr = syscall_nr;
    event->fd = fd;
    event->size = size;
    event->offset = offset;
    if (owner)
        __builtin_memcpy(event->comm, owner->comm, sizeof(event->comm));
    else
        bpf_get_current_comm(&event->comm, sizeof(event->comm));
    __builtin_memcpy(event->filename, filename , sizeof(event->filename));
    event->open_flags_hex = open_flags_hex;
    __builtin_memcpy(event->open_flags_str, open_flags_str, sizeof(event->open_flags_str));
    if (syscall_nr == 0 || syscall_nr == 17)
        event->ddir = READ;
    else if (syscall_nr == 1 || syscall_nr == 18)
        event->ddir = WRITE;
    else if (syscall_nr == 19)
        event->ddir = VREAD;
    else if (syscall_nr == 20)
        event->ddir = VWRITE;
    else if (syscall_nr == EVENT_NR_FAULT_BIO || syscall_nr == EVENT_NR_FAULT_RQ_COMPLETE)
        event->ddir = (open_flags_hex & 0xff) == 1 ? WRITE : READ;
    event->ret = ret;
    event->error_code = error_code;

    bpf_ringbuf_submit(event, 0);
}

static __always_inline void log_event(u32 syscall_nr, u32 fd, u64 size, s64 offset, char filename[256], int open_flags_hex, char open_flags_str[128], long ret, long error_code)
{
    log_event_with_owner(syscall_nr, fd, size, offset, filename, open_flags_hex,
                         open_flags_str, ret, error_code, NULL);
}

// read syscall tracepoint
SEC("kprobe/__x64_sys_read")
int trace_read_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 0; // read
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;
    unsigned int fd   = (unsigned int)PT_REGS_PARM1(p);  /* rdi ✓ */
    size_t       size = (size_t)PT_REGS_PARM3(p);        /* rdx ✓ */
    /* read has no offset — positional read uses file position */

    update_stats(syscall_nr, size);
    log_event(syscall_nr, fd, size, 0, "", 0, "", -1,-1);

    return 0;
}

// write syscall tracepoint
SEC("kprobe/__x64_sys_write")
int trace_write_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 1; // write
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;
    unsigned int fd   = (unsigned int)PT_REGS_PARM1(p);  /* rdi ✓ */
    size_t       size = (size_t)PT_REGS_PARM3(p);        /* rdx ✓ */
    /* write has no offset — positional write uses file position */

    update_stats(syscall_nr, size);
    log_event(syscall_nr, fd, size, 0, "", 0, "", -1, -1);

    return 0;
}

SEC("kprobe/__x64_sys_open")
int trace_open_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 2; // open

    update_stats(syscall_nr, 1);
    log_event(syscall_nr, -1, 1, 0, "", 0, "", -1, -1);

    return 0;
}

//openat syscall tracepoint
SEC("tracepoint/syscalls/sys_enter_openat")
int trace_openat(struct trace_event_raw_sys_enter *ctx)
{
    char filename[256];
    u32 syscall_nr = 257;
    int dfd = (int)ctx->args[0];
    const char *filenameptr = (const char *)ctx->args[1];
    uint32_t open_flags_hex = (uint32_t)ctx->args[2];

    if (bpf_probe_read_user_str(filename, sizeof(filename), filenameptr) < 0) {
        bpf_printk("failed to read filename\n");
        return 0;
    }

    char open_flags_str[128] = {0};

    int accmode  = open_flags_hex & O_ACCMODE;

    /*decode the hex flags for file access modes */
    if (accmode == O_RDONLY) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_RDONLY", NULL, 0);
    } else if (accmode == O_WRONLY) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_WRONLY", NULL, 0);
    } else if (accmode == O_RDWR) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_RDWR", NULL, 0);
    }


    if (open_flags_hex & O_CREAT) {
        bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_CREAT", NULL, 0);
    }
    if (open_flags_hex & O_EXCL) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_EXCL", NULL, 0);
    }
    if (open_flags_hex & O_NOCTTY) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_NOCTTY", NULL, 0);
    }
    if (open_flags_hex & O_TRUNC) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_TRUNC", NULL, 0);
    }
    if (open_flags_hex & O_APPEND) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_APPEND", NULL, 0);
    }
    if (open_flags_hex & O_NONBLOCK) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_NONBLOCK", NULL, 0);
    }
    if (open_flags_hex & O_DSYNC) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_DSYNC", NULL, 0);
    }
    if (open_flags_hex & FASYNC) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "FASYNC", NULL, 0);
    }
    if (open_flags_hex & O_DIRECT) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_DIRECT", NULL, 0);
    }
    if (open_flags_hex & O_LARGEFILE) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_LARGEFILE", NULL, 0);
    }
    if (open_flags_hex & O_DIRECTORY) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_DIRECTORY", NULL, 0);
    }
    if (open_flags_hex & O_NOFOLLOW) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_NOFOLLOW", NULL, 0);
    }
    if (open_flags_hex & O_CLOEXEC) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_CLOEXEC", NULL, 0);
    }
    if (open_flags_hex & O_SYNC) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_SYNC", NULL, 0);
    }
    if (open_flags_hex & O_PATH) {
       bpf_snprintf(open_flags_str, sizeof(open_flags_str), "O_PATH", NULL, 0);
    }

    update_stats(syscall_nr, 1);
    log_event(syscall_nr, dfd, 1, 0, filename, open_flags_hex, open_flags_str, -1, -1);

    return 0;
}

// close syscall tracepoint
SEC("kprobe/__x64_sys_close")
int trace_close_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 3; // close
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;
    struct pt_regs *p = &inner;

    unsigned int fd = (unsigned int)PT_REGS_PARM1(p);  /* rdi */

    update_stats(syscall_nr, 1);
    log_event(syscall_nr, fd, 1, 0, "", 0, "", -1, -1);

    return 0;
}

// lseek syscall tracepoint
SEC("kprobe/__x64_sys_lseek")
int trace_lseek_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 8; // lseek
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;

    unsigned int fd     = (unsigned int)PT_REGS_PARM1(p);  /* rdi ✓ */
    s64        offset = (s64)PT_REGS_PARM2(p);         /* rsi ✓ */
    unsigned int whence = (unsigned int)PT_REGS_PARM3(p);  /* rdx ✓ */

    u64 abs_offset = offset > 0 ? (u64)offset : (u64)-offset;

    update_stats(syscall_nr, abs_offset);
    log_event(syscall_nr, fd, abs_offset, (loff_t)offset, "", whence, "", -1, -1);

    return 0;
}

// pread64 syscall tracepoint
SEC("kprobe/__x64_sys_pread64")
int trace_pread_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 17; // pread64
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;

    unsigned int fd     = (unsigned int)PT_REGS_PARM1(p);  /* rdi ✓ */
    u64          buf    = (u64)PT_REGS_PARM2(p);           /* rsi — buf ptr, not logged */
    size_t       size   = (size_t)PT_REGS_PARM3(p);        /* rdx ✓ */

    /* pread64 arg4 = offset uses r10 not rcx — same as mmap arg4 */
    loff_t offset = (loff_t)inner.r10;

    update_stats(syscall_nr, size);
    log_event(syscall_nr, fd, size, offset, "", 0, "", -1, -1);

    return 0;
}

// pwrite64 syscall tracepoint
SEC("kprobe/__x64_sys_pwrite64")
int trace_pwrite_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 18; // pwrite64
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;

    unsigned int fd     = (unsigned int)PT_REGS_PARM1(p);
    size_t       size   = (size_t)PT_REGS_PARM3(p);
    loff_t       offset = (loff_t)inner.r10;   /* arg4 = r10 ✓ */

    update_stats(syscall_nr, size);
    log_event(syscall_nr, fd, size, offset, "", 0, "", -1, -1);

    return 0;
}

// mmap syscall tracepoint
SEC("kprobe/__x64_sys_mmap")
int trace_mmap_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 9; // mmap syscall
    /*
     * Safely read the inner pt_regs pointer via bpf_probe_read_kernel.
     * Direct cast of PT_REGS_PARM1(ctx) gives a scalar the verifier
     * won't allow dereferencing — read it through the helper instead.
     */
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    /* now read args from the local copy — all lvalues, verifier happy */
    unsigned int fd        = (unsigned int)PT_REGS_PARM5(&inner);
    size_t       length    = (size_t)PT_REGS_PARM2(&inner);
    loff_t       offset    = (loff_t)PT_REGS_PARM6(&inner);
    int          prot      = (int)PT_REGS_PARM3(&inner);
    int          map_flags = (int)inner.r10;
    char open_flags_str[128] = {};
    __u32 pos    = 0;
    int   need_sep = 0;

     /*
     * Pack prot and map_flags as "PROT:FLAGS" hex string — tiny, fixed size,
     * no unrolled loops. Decoded in userspace parse_mmap_flags().
     * e.g. prot=3, map_flags=2 → "0x00000003:0x00000002"
     */

    /* emit "0x" + 8 hex digits for prot */
    const char hex[] = "0123456789abcdef";
    open_flags_str[pos++] = '0';
    open_flags_str[pos++] = 'x';
    #pragma unroll
    for (int i = 7; i >= 0; i--)
        open_flags_str[pos++] = hex[((unsigned)prot >> (i * 4)) & 0xF];

    open_flags_str[pos++] = ':';

    /* emit "0x" + 8 hex digits for map_flags */
    open_flags_str[pos++] = '0';
    open_flags_str[pos++] = 'x';
    #pragma unroll
    for (int i = 7; i >= 0; i--)
        open_flags_str[pos++] = hex[((unsigned)map_flags >> (i * 4)) & 0xF];

    open_flags_str[pos] = '\0';
    /* result: "0x00000003:0x00000002" — 21 bytes, fixed, verifier-friendly */

    open_flags_str[pos] = '\0';

    update_stats(syscall_nr, length);
    log_event(syscall_nr, fd, length, offset, "", 0, open_flags_str, -1, -1);

    return 0;
}

/*mmap exit syscall tracepoint */
SEC("kretprobe/__x64_sys_mmap")
int trace_mmap_exit(struct pt_regs *ctx)
{
    u32 syscall_nr = 9; // mmap syscall
    unsigned long ret_addr = (unsigned long)PT_REGS_RC(ctx);

    char filename[256] = {};
    if (ret_addr == (unsigned long)-1UL) {
        filename[0] = '0'; filename[1] = 'x'; filename[2] = '0'; filename[3] = '\0';
    } else {
        const char hex[] = "0123456789abcdef";
        filename[0] = '0';
        filename[1] = 'x';
        #pragma unroll
        for (int i = 0; i < 16; i++)
            filename[2 + i] = hex[(ret_addr >> ((15 - i) * 4)) & 0xF];
        filename[18] = '\0';
    }

    update_stats(syscall_nr, 0);
    log_event(syscall_nr, 0, 0, 0, filename, 0, "", -1, -1);

    return 0;
}

//munmap syscall tracepoint
SEC("kprobe/__x64_sys_munmap")
int trace_munmap_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 11;
    /* read only the two registers we need — not full pt_regs */
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);

    unsigned long addr   = 0;
    size_t        length = 0;

    /* read addr (rdi = offset 112) and length (rsi = offset 104) directly */
    bpf_probe_read_kernel(&addr,   sizeof(addr),   &inner_ptr->di);
    bpf_probe_read_kernel(&length, sizeof(length),  &inner_ptr->si);

    /* encode addr as hex into filename[] — replayer uses it as cap_addr key */
    char filename[256] = {};
    const char hex[] = "0123456789abcdef";
    filename[0] = '0';
    filename[1] = 'x';
    #pragma unroll
    for (int i = 0; i < 16; i++)
        filename[2 + i] = hex[(addr >> ((15 - i) * 4)) & 0xF];
    filename[18] = '\0';

    update_stats(syscall_nr, length);
    log_event(syscall_nr, -1, length, 0, filename, 0, "", -1, -1);

    return 0;
}

// readv syscall tracepoint
SEC("kprobe/__x64_sys_readv")
int trace_readv_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 19;
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;
    unsigned int fd    = (unsigned int)PT_REGS_PARM1(p);  /* rdi */
    unsigned long count = (unsigned long)PT_REGS_PARM3(p); /* rdx = iovcnt */

    update_stats(syscall_nr, count);
    log_event(syscall_nr, fd, count, 0, "", 0, "", -1, -1);

    return 0;
}

// writev syscall tracepoint
SEC("kprobe/__x64_sys_writev")
int trace_writev_entry(struct pt_regs *ctx)
{
    u32 syscall_nr = 20; // writev
    struct pt_regs inner = {};
    struct pt_regs *inner_ptr = (struct pt_regs *)PT_REGS_PARM1(ctx);
    if (bpf_probe_read_kernel(&inner, sizeof(inner), inner_ptr) < 0)
        return 0;

    struct pt_regs *p = &inner;
    unsigned int fd    = (unsigned int)PT_REGS_PARM1(p);  /* rdi ✓ */
    unsigned long count = (unsigned long)PT_REGS_PARM3(p); /* rdx = iovcnt ✓ */

    update_stats(syscall_nr, count);
    log_event(syscall_nr, fd, count, 0, "", 0, "", -1, -1);

    return 0;
}

SEC("tracepoint/syscalls/sys_enter_fsync")
int trace_fsync_entry(struct trace_event_raw_sys_enter *ctx)
{
    u32 syscall_nr = 74; // fsync
    unsigned int fd = (unsigned int)ctx->args[0];

    update_stats(syscall_nr, 1);
    log_event(syscall_nr, fd, 1, 0, "", 0, "", -1, -1);

    return 0;
}

/* ====================================================================
 * LAYER 1 — handle_mm_fault(): file-backed user faults
 *
 * Why handle_mm_fault and not filemap_fault:
 *   - filesystem-agnostic (filemap_fault is the generic path, but
 *     some filesystems supply their own ->fault)
 *   - gives the `flags` argument, so FAULT_FLAG_USER can be filtered
 *   - gives the vma directly, so the file offset comes from the
 *     kernel's own bookkeeping instead of a userspace region table
 *
 * Signature:
 *   vm_fault_t handle_mm_fault(struct vm_area_struct *vma,
 *                              unsigned long address,
 *                              unsigned int flags,
 *                              struct pt_regs *regs);
 * ==================================================================== */

/* per-task fault context: key = pid_tgid */
struct {
    __uint(type,        BPF_MAP_TYPE_HASH);
    __uint(max_entries, FAULT_CTX_MAX);
    __type(key,         u64);
    __type(value,       struct fault_ctx);
} fault_ctx_map SEC(".maps");

/* 256-byte filename scratch, per-CPU so it costs no BPF stack */
struct {
    __uint(type,        BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key,         u32);
    __type(value,       struct fault_strbuf);
} fault_strbuf_map SEC(".maps");

static __always_inline struct fault_strbuf *fault_strbuf_get(void)
{
    u32 k = 0;
    return bpf_map_lookup_elem(&fault_strbuf_map, &k);
}

/*
 * Resolve the backing filename from vm_file.
 * Returns 0 on success, -1 if the dentry chain could not be read.
 */
static __always_inline int
fault_read_filename(struct file *vm_file, char *dst, int dst_sz)
{
    struct dentry *dentry = BPF_CORE_READ(vm_file, f_path.dentry);
    if (!dentry)
        return -1;

    const unsigned char *name = BPF_CORE_READ(dentry, d_name.name);
    if (!name)
        return -1;

    if (bpf_probe_read_kernel_str(dst, dst_sz, name) <= 0)
        return -1;

    return 0;
}
/* borrowed from mm.h: folio_test_large() == folio_test_head() */
static __always_inline bool folio_test_large_safe(struct folio *folio)
{
    unsigned long flags = BPF_CORE_READ(folio, flags.f);
    return (flags & (1UL << PG_head)) != 0;
}

/* borrowed from mm.h: folio_large_nr_pages() == folio->_nr_pages
 */
static __always_inline __u32 folio_large_nr_pages_safe(struct folio *folio)
{
    if (bpf_core_field_exists(folio->_nr_pages)) {
        __u32 n = BPF_CORE_READ(folio, _nr_pages);
        if (n > 0 && n <= 512)        /* bound keeps the verifier happy */
            return n;
    }

    /* field removed: derive from order */
    __u32 order = (__u32)(BPF_CORE_READ(folio, _flags_1) & 0xff);
    if (order > 9)                      /* cap at 2 MB */
        order = 9;
    return 1U << order;
}

/* number of pages covered in a folio , for both single page and large folios */
static __always_inline __u32 folio_nr_pages_safe(struct folio *folio)
{
    if (!folio_test_large_safe(folio))
        return 1;
    return folio_large_nr_pages_safe(folio);
}


SEC("fentry/handle_mm_fault")
int BPF_PROG(trace_handle_mm_fault_enter,
             struct vm_area_struct *vma,
             unsigned long address,
             unsigned int flags,
             struct pt_regs *regs)
{

    /* filter faults that originated from userspace.
     */
    if (!(flags & FAULT_FLAG_USER))
        return 0;

    /* vm_file == NULL  =>  anonymous: heap, stack, MAP_ANONYMOUS,
     * CUDA/GPU buffers.  These never read from a filesystem, so a
     * major fault on them means swap, which is not workload IO.
     */
    struct file *vm_file = BPF_CORE_READ(vma, vm_file);
    if (!vm_file)
        return 0;

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid     = (u32)(pid_tgid >> 32);
    if (!is_target_pid(tgid))
        return 0;


    u64 vm_start = BPF_CORE_READ(vma, vm_start); /* used to de-reference kernel data structure. */
    u64 vm_pgoff = BPF_CORE_READ(vma, vm_pgoff); /* This is the page aligned offset within a file that is memory mapped;*/

    if (address < vm_start)
        return 0;

    /* file offset, straight from the kernel's VMA */
    u64 file_offset = (vm_pgoff << PAGE_SHIFT) + (address - vm_start);

    struct inode *inode = BPF_CORE_READ(vm_file, f_inode);
    u64 ino = inode ? BPF_CORE_READ(inode, i_ino) : 0;
    u32 dev = inode ? BPF_CORE_READ(inode, i_sb, s_dev) : 0;

    struct fault_ctx fc = {};
    fc.address     = address;
    fc.file_offset = file_offset;
    fc.ino         = ino;
    fc.dev         = dev;
    fc.fault_flags = flags;
    fc.start_ns    = bpf_ktime_get_ns();
    fc.vm_start    = vm_start;
    fc.vm_pgoff    = vm_pgoff;
    fc.ra_pages    = 0;
    fc.folios_added = 0;
    fc.l2_logged   = 0;
    fc.bio_count   = 0;
    fc.bio_bytes   = 0;
    fc.in_fault    = 1;

    bpf_map_update_elem(&fault_ctx_map, &pid_tgid, &fc, BPF_ANY);

    return 0;
}

SEC("fexit/handle_mm_fault")
int BPF_PROG(trace_handle_mm_fault_exit,
             struct vm_area_struct *vma,
             unsigned long address,
             unsigned int flags,
             struct pt_regs *regs,
             vm_fault_t ret)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc)
        return 0;

    u32 r = (u32)ret;

    /*
     * VM_FAULT_RETRY handling.
     *
     * On a cold-cache fault the kernel typically drops the mmap lock to
     * perform IO and returns MAJOR|RETRY on the first pass; the retried
     * call then finds the page resident and returns a plain minor.
     * Checking MAJOR on every exit therefore yields exactly one record
     * per fault, matching how the kernel's own ru_majflt is accounted.
     *
     * The entry probe re-populates the context on the retry, so the
     * unconditional delete below is safe.
     */
    int is_major = (r & VM_FAULT_MAJOR) != 0;
    int is_error = (r & VM_FAULT_ERROR) != 0;

    if (is_major && !is_error) {
        struct fault_strbuf *sb = fault_strbuf_get();
        char *fname = sb ? sb->buf : NULL;
        if (fname) {
            __builtin_memset(fname, 0, sizeof(sb->buf));
            struct file *vm_file = BPF_CORE_READ(vma, vm_file);
            if (vm_file)
                fault_read_filename(vm_file, fname, sizeof(sb->buf));
        }

        u64 latency_ns = bpf_ktime_get_ns() - fc->start_ns;

        update_stats(EVENT_NR_PAGE_FAULT_MAJOR, PAGE_SIZE);

        log_event(EVENT_NR_PAGE_FAULT_MAJOR,
                  -1,                        /* fd: none for a fault     */
                  PAGE_SIZE,                 /* size: one page           */
                  fc->file_offset,           /* offset: file byte offset */
                  fname ? fname : "",        /* backing file             */
                  (int)r,                    /* vm_fault_t return value  */
                  "",                        /* flags string unused      */
                  (long)latency_ns,          /* ret: fault latency       */
                  fc->ra_pages);             /* errno slot: reserved     */
    }

    bpf_map_delete_elem(&fault_ctx_map, &pid_tgid);
    return 0;
}

/* ====================================================================
 * LAYER 2 — filesystem readahead / folio read
 * ext4_readahead()  gives the window size via readahead_control
 * ext4_read_folio() covers the single-folio path when RA does not fire
 *
 * Both run in the faulting task's context when driven synchronously by
 * a fault, so the pid_tgid lookup into fault_ctx_map works directly.
 * ==================================================================== */

/* ── ext4 ─────────────────────────────────────────────────────────── */

SEC("fentry/ext4_readahead")
int BPF_PROG(trace_ext4_readahead, struct readahead_control *rac)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;                 /* readahead outside a tracked fault */

    if (fc->l2_logged)
        return 0;
    u32 nr_pages = BPF_CORE_READ(rac, _nr_pages); /* number of pages read ahead done */
    u64 index    = BPF_CORE_READ(rac, _index); /* page index of the first page that is present in the readahead window. */

    fc->ra_pages = nr_pages;
    fc->l2_logged = 1;

    update_stats(EVENT_NR_FAULT_READAHEAD, nr_pages * PAGE_SIZE);

    log_event(EVENT_NR_FAULT_READAHEAD,
              -1,
              (long)(nr_pages * PAGE_SIZE),
              index << PAGE_SHIFT,
              "",
              (int)nr_pages,
              "",
              (long)fc->file_offset,
              0);

    return 0;
}

SEC("fentry/ext4_read_folio")
int BPF_PROG(trace_ext4_read_folio, struct file *file, struct folio *folio)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;
    if (fc->l2_logged)
        return 0;
    /* ext4_readahead already accounted this fault's fetch; both
     * callbacks can fire within one fault and would double-count */
    if (fc->ra_pages != 0)
        return 0;

    __u32 nr_pages = folio_nr_pages_safe(folio);
    __u32 bytes    = nr_pages * PAGE_SIZE_BPF;

    /* folio->index is the page offset of this folio within the file */
    __u64 index = BPF_CORE_READ(folio, index);

    fc->ra_pages = nr_pages;
    fc->l2_logged = 1;

    update_stats(EVENT_NR_FAULT_READAHEAD, bytes);

    /* same record shape as the readahead path, so both appear
     * identically in the JSON and can be summed together        */
    log_event(EVENT_NR_FAULT_READAHEAD,
              -1,
              (long)bytes,              /* bytes this fault fetched  */
              index << PAGE_SHIFT,      /* file byte offset of folio */
              "",
              (int)nr_pages,            /* pages, for symmetry       */
              "",
              (long)fc->file_offset,    /* join key back to L1       */
              0);
    return 0;
}

/* ── xfs ──────────────────────────────────────────────────────────── *
 *
 * Attach only if the target files live on XFS. libbpf reports a
 * missing-symbol error at load for whichever filesystem is absent, so
 * this program is disabled via bpf_program__set_autoload() by the
 * loader unless xfs is the detected backing filesystem.
 */
SEC("fentry/xfs_vm_readahead")
int BPF_PROG(trace_xfs_readahead, struct readahead_control *rac)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;

    u32 nr_pages = BPF_CORE_READ(rac, _nr_pages);
    u64 index    = BPF_CORE_READ(rac, _index);
    fc->ra_pages = nr_pages;

    update_stats(EVENT_NR_FAULT_READAHEAD, nr_pages * PAGE_SIZE);
    log_event(EVENT_NR_FAULT_READAHEAD, -1,
              (long)(nr_pages * PAGE_SIZE),
              index << PAGE_SHIFT, "", (int)nr_pages, "",
              (long)fc->file_offset, 0); /*offset here is logged into fault_ctx during handle_mm_fault probe */
    return 0;
}

/* ── filesystem-agnostic fallback ─────────────────────────────────── *
 *
 * mm/filemap.c: filemap_add_folio() is called once per folio actually
 * inserted into the page cache, regardless of filesystem. This is the
 * single best "real pages fetched" signal if per-fs instrumentation is
 * not practical.
 */
SEC("fentry/filemap_add_folio")
int BPF_PROG(trace_filemap_add_folio,
             struct address_space *mapping, struct folio *folio,
             unsigned long index, unsigned int gfp)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;

    /* confirm this folio belongs to the faulting file */
    struct inode *host = BPF_CORE_READ(mapping, host);
    if (!host)
        return 0;
    u64 ino = BPF_CORE_READ(host, i_ino);
    if (ino != fc->ino)
        return 0;                 /* different file - not our fault's IO */

    fc->folios_added += 1;
    return 0;
}

/* ====================================================================
 * LAYER 3 — fault BIO submission and request completion
 *
 * a major fault blocks the faulting task until its IO completes, so any bio
 * that task submits between handle_mm_fault entry and exit is causally
 * that fault's IO, and L1 already recorded which file and offset the
 * fault was for.
 * Request attribution requires issue in the active faulting task.
 * Plug flushes and delayed dispatch in other contexts are not attributed.
 * ==================================================================== */
struct {
    __uint(type,        BPF_MAP_TYPE_HASH);
    __uint(max_entries, FAULT_CTX_MAX);
    __type(key,         __u64);
    __type(value,       struct fault_request_ctx);
} fault_request_map SEC(".maps");

SEC("fentry/submit_bio")
int BPF_PROG(trace_submit_bio, struct bio *bio)
{
    if (!bio)
        return 0;

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;

    __u64 sector = BPF_CORE_READ(bio, bi_iter.bi_sector);
    __u32 bytes  = BPF_CORE_READ(bio, bi_iter.bi_size);
    __u32 opf    = BPF_CORE_READ(bio, bi_opf);
    if (bytes == 0)
        return 0;

    __u64 fault_off = fc->file_offset;
    fc->bio_count += 1;
    fc->bio_bytes += bytes;

    update_stats(EVENT_NR_FAULT_BIO, bytes);

    log_event(EVENT_NR_FAULT_BIO,
              -1,
              (long)bytes,
              sector << 9,
              "",
              (int)opf,
              "",
              (long)fault_off,
              1);
    return 0;
}

SEC("tp_btf/block_rq_issue")
int BPF_PROG(trace_block_rq_issue, struct request *rq)
{
    if (!rq)
        return 0;

    __u64 request_key = (__u64)rq;
    if (bpf_map_lookup_elem(&fault_request_map, &request_key))
        return 0;

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_ctx *fc = bpf_map_lookup_elem(&fault_ctx_map, &pid_tgid);
    if (!fc || !fc->in_fault)
        return 0;

    struct fault_request_ctx request = {};
    request.total_bytes = BPF_CORE_READ(rq, __data_len);
    if (!request.total_bytes)
        return 0;
    request.owner.pid_tgid = pid_tgid;
    bpf_get_current_comm(request.owner.comm, sizeof(request.owner.comm));
    request.issue_ns = bpf_ktime_get_ns();
    request.file_offset = fc->file_offset;
    request.sector = BPF_CORE_READ(rq, __sector);
    request.opf = BPF_CORE_READ(rq, cmd_flags); /* op and common flags */
    bpf_map_update_elem(&fault_request_map, &request_key, &request, BPF_NOEXIST);
    return 0;
}

SEC("tp_btf/block_rq_complete")
int BPF_PROG(trace_block_rq_complete, struct request *rq,
             blk_status_t error, unsigned int nr_bytes)
{
    __u64 request_key = (__u64)rq;
    struct fault_request_ctx *request = bpf_map_lookup_elem(&fault_request_map, &request_key);
    if (!request)
        return 0;

    if (!fault_request_account_completion(request, nr_bytes, error))
        return 0;

    __u64 latency_ns = bpf_ktime_get_ns() - request->issue_ns;
    update_stats(EVENT_NR_FAULT_RQ_COMPLETE, request->completed_bytes);

    log_event_with_owner(EVENT_NR_FAULT_RQ_COMPLETE, -1,
                         request->completed_bytes, request->sector << 9,
                         "", (int)request->opf, "", (long)latency_ns,
                         request->status, &request->owner);

    bpf_map_delete_elem(&fault_request_map, &request_key);
    return 0;
}

// syscall exit tracepoint
SEC("tracepoint/raw_syscalls/sys_exit")
int trace_sys_exit(struct trace_event_raw_sys_exit *ctx)
{
    u32 syscall_nr  = ctx->id;   // syscall number
    long ret = ctx->ret; // return value
    long error_code  = (ret < 0) ? ret : 0;   /* negative errno or 0 */

    /* filter to tracked syscalls only */
    if (syscall_nr != 0  && syscall_nr != 1  && syscall_nr != 2  &&
        syscall_nr != 3  && syscall_nr != 8  && syscall_nr != 9  &&
        syscall_nr != 11 && syscall_nr != 17 && syscall_nr != 18 &&
        syscall_nr != 19 && syscall_nr != 20 && syscall_nr != 74 &&
        syscall_nr != 257)
        return 0;

    /*
     * Exit record — fd=-1, size=1, offset=0 signals to the replayer
     * that this is an exit record and should be skipped.
     * Contains only ret and error_code — no syscall args.
     */
    log_event(syscall_nr, -1, 1, 0, "", 0, "", ret, error_code);

    return 0;
}

char _license[] SEC("license") = "GPL";
