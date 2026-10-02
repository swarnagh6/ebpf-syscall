/** ====================================================================
 * fault_io_tracking.bpf.h
 *
 * Three-layer page-fault -> IO attribution for file-backed mmap
 * regions
 *
 *   L1  handle_mm_fault()   - identify file-backed user faults, derive
 *                             the file offset from the VMA itself
 *   L2  <fs>_readahead()    - how many pages the fault actually fetched
 *       <fs>_read_folio()
 *   L3  submit_bio()        - real sectors/bytes reaching the device
 *       block_rq_issue()    - attribute requests in the faulting task
 *       block_rq_complete() - accumulate completed request bytes
 *
 * Each layer is independently useful; L1 alone already supersedes a
 * userspace mmap-region table because vma->vm_pgoff is authoritative
 * and needs no prior observation of the mmap() call.
 * ==================================================================== */
#ifndef __FAULT_IO_TRACKING_BPF_H
#define __FAULT_IO_TRACKING_BPF_H
#ifndef PAGE_SHIFT
#define PAGE_SHIFT              12
#endif
#ifndef PAGE_SIZE_BPF
#define PAGE_SIZE_BPF           4096
#endif

/* include/linux/mm_types.h : enum fault_flag */
#define FAULT_FLAG_WRITE        0x01
#define FAULT_FLAG_MKWRITE      0x02
#define FAULT_FLAG_ALLOW_RETRY  0x04
#define FAULT_FLAG_RETRY_NOWAIT 0x08
#define FAULT_FLAG_KILLABLE     0x10
#define FAULT_FLAG_TRIED        0x20
#define FAULT_FLAG_USER         0x40
#define FAULT_FLAG_REMOTE       0x80
#define FAULT_FLAG_INSTRUCTION  0x100
#define FAULT_FLAG_INTERRUPTIBLE 0x200

/* include/linux/mm_types.h : enum vm_fault_reason */
#define VM_FAULT_OOM            0x000001
#define VM_FAULT_SIGBUS         0x000002
#define VM_FAULT_MAJOR          0x000004
#define VM_FAULT_HWPOISON       0x000010
#define VM_FAULT_SIGSEGV        0x000040
#define VM_FAULT_NOPAGE         0x000100
#define VM_FAULT_LOCKED         0x000200
#define VM_FAULT_RETRY          0x000400
#define VM_FAULT_FALLBACK       0x000800
#define VM_FAULT_ERROR  (VM_FAULT_OOM | VM_FAULT_SIGBUS | VM_FAULT_SIGSEGV | \
                         VM_FAULT_HWPOISON | VM_FAULT_FALLBACK)
/* include/linux/page-flags.h : folio->mapping low bit tags an anon folio */
#define FOLIO_MAPPING_ANON      0x1

/* event ids emitted into the UTF JSON stream */
#define EVENT_NR_PAGE_FAULT_MAJOR   1025   /* L1: major file-backed fault  */
#define EVENT_NR_FAULT_READAHEAD    1026   /* L2: pages pulled by the fault*/
#define EVENT_NR_FAULT_BIO          1027   /* L3: bio issued for the fault */
#define EVENT_NR_FAULT_RQ_COMPLETE  1028
#define FAULT_CTX_MAX               10240

/* ── per-task fault context ───────────────────────────────────────── *
 * A task can only be inside one fault at a time (it is blocked until
 * the fault resolves), so pid_tgid alone is a sufficient key.  This is
 * also what makes L3 tractable: any bio this task submits between
 * entry and exit is causally this fault's IO, with no sector->file
 * reverse mapping required.
 * ------------------------------------------------------------------ */
struct fault_ctx {
    __u64 address;        /* faulting virtual address                 */
    __u64 file_offset;    /* (vm_pgoff << PAGE_SHIFT) + (addr - start)*/
    __u64 ino;            /* backing inode - correlation key for L3   */
    __u32 dev;            /* backing device                           */
    __u32 fault_flags;    /* FAULT_FLAG_* as passed in                */
    __u64 start_ns;       /* fault entry timestamp                    */
    __u64 vm_start;       /* VMA snapshot: can be split/unmapped later*/
    __u64 vm_pgoff;
    __u32 ra_pages;       /* L2: pages the readahead window covered   */
    __u32 folios_added;   /* represents  filemap_add_folio count      */
    __u8  l2_logged;       /* 1 if L2 readahead was logged */
    __u32 bio_count;      /* L3: bios seen inside this fault          */
    __u64 bio_bytes;      /* L3: total bytes submitted                */
    __u8  in_fault;       /* 1 while inside handle_mm_fault           */
};

/* scratch for the filename, kept off the 512-byte BPF stack */
struct fault_strbuf {
    char buf[256];
};

struct fault_io_owner {
    __u64 pid_tgid;
    char comm[16];
};

struct fault_request_ctx {
    struct fault_io_owner owner;
    __u64 issue_ns;
    __u64 file_offset;
    __u64 sector;
    __u32 total_bytes;
    __u32 completed_bytes;
    __u32 opf;
    __u32 status;
};

static inline int fault_request_account_completion(struct fault_request_ctx *request,
                                                   __u32 nr_bytes, __u32 status)
{
    __u32 remaining = request->total_bytes - request->completed_bytes;
    request->completed_bytes += nr_bytes > remaining ? remaining : nr_bytes;
    if (status && !request->status)
        request->status = status;
    return request->completed_bytes == request->total_bytes; // return if the accumulated bytes reach the total requested.
}

#endif /* __FAULT_IO_TRACKING_BPF_H */