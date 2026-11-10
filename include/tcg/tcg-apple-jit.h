/*
 * Apple Silicon functions for JIT handling
 *
 * Copyright (c) 2020 osy
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TCG_APPLE_JIT_H
#define TCG_APPLE_JIT_H

/*
 * APRR handling
 * Credits to: https://siguza.github.io/APRR/
 * Reversed from /usr/lib/system/libsystem_pthread.dylib
 */

#if defined(__aarch64__) && defined(CONFIG_DARWIN)

#define _COMM_PAGE_START_ADDRESS        (0x0000000FFFFFC000ULL) /* In TTBR0 */
#define _COMM_PAGE_APRR_SUPPORT         (_COMM_PAGE_START_ADDRESS + 0x10C)
#define _COMM_PAGE_APPR_WRITE_ENABLE    (_COMM_PAGE_START_ADDRESS + 0x110)
#define _COMM_PAGE_APRR_WRITE_DISABLE   (_COMM_PAGE_START_ADDRESS + 0x118)

static __attribute__((__always_inline__)) bool jit_write_protect_supported(void)
{
    /* Access shared kernel page at fixed memory location. */
    uint8_t aprr_support = *(volatile uint8_t *)_COMM_PAGE_APRR_SUPPORT;
    return aprr_support > 0;
}

/* write protect enable = write disable */
static __attribute__((__always_inline__)) void jit_write_protect(int enabled)
{
    /* Access shared kernel page at fixed memory location. */
    uint8_t aprr_support = *(volatile uint8_t *)_COMM_PAGE_APRR_SUPPORT;
    if (aprr_support == 0 || aprr_support > 3) {
        return;
    } else if (aprr_support == 1) {
        __asm__ __volatile__ (
            "mov x0, %0\n"
            "ldr x0, [x0]\n"
            "msr S3_4_c15_c2_7, x0\n"
            "isb sy\n"
            :: "r" (enabled ? _COMM_PAGE_APRR_WRITE_DISABLE
                            : _COMM_PAGE_APPR_WRITE_ENABLE)
            : "memory", "x0"
        );
    } else {
        __asm__ __volatile__ (
            "mov x0, %0\n"
            "ldr x0, [x0]\n"
            "msr S3_6_c15_c1_5, x0\n"
            "isb sy\n"
            :: "r" (enabled ? _COMM_PAGE_APRR_WRITE_DISABLE
                            : _COMM_PAGE_APPR_WRITE_ENABLE)
            : "memory", "x0"
        );
    }
}

#else /* defined(__aarch64__) && defined(CONFIG_DARWIN) */

static __attribute__((__always_inline__)) bool jit_write_protect_supported(void)
{
    return false;
}

static __attribute__((__always_inline__)) void jit_write_protect(int enabled)
{
}

#endif

/*
 * Switching the JIT mapping between writable and executable costs an
 * "isb sy" -- a full pipeline resynchronisation.  qemu_thread_jit_execute()
 * runs before every translated block is entered, while writes happen only
 * when code is generated or patched, so remember which state this thread is
 * already in and skip the switch (and the barrier) when nothing changes.
 * The APRR state is per-thread, so the cache is too -- and it must be a
 * single variable shared by every translation unit, or one file's idea of
 * the state would let another skip a switch it actually needed.
 */
extern __thread int qemu_jit_wx_state;

static inline void qemu_thread_jit_execute(void)
{
    if (qemu_jit_wx_state != 1 && jit_write_protect_supported()) {
        jit_write_protect(true);
        qemu_jit_wx_state = 1;
    }
}

static inline void qemu_thread_jit_write(void)
{
    if (qemu_jit_wx_state != 0 && jit_write_protect_supported()) {
        jit_write_protect(false);
        qemu_jit_wx_state = 0;
    }
}

#endif /* define TCG_APPLE_JIT_H */
