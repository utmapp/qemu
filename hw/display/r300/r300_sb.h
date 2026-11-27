/*
 * Tiny growable string builder used by the R300 shader translators.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#ifndef HW_DISPLAY_R300_SB_H
#define HW_DISPLAY_R300_SB_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct R300Sb {
    char *buf;
    size_t len;
    size_t cap;
} R300Sb;

static inline void r300_sb_init(R300Sb *sb)
{
    sb->cap = 4096;
    sb->len = 0;
    sb->buf = malloc(sb->cap);
    sb->buf[0] = '\0';
}

static inline void r300_sb_free(R300Sb *sb)
{
    free(sb->buf);
    sb->buf = NULL;
    sb->len = sb->cap = 0;
}

static inline void r300_sb_vprintf(R300Sb *sb, const char *fmt, va_list ap)
{
    for (;;) {
        va_list aq;
        va_copy(aq, ap);
        int n = vsnprintf(sb->buf + sb->len, sb->cap - sb->len, fmt, aq);
        va_end(aq);
        if (n < 0) {
            return;
        }
        if ((size_t)n < sb->cap - sb->len) {
            sb->len += n;
            return;
        }
        sb->cap = (sb->cap + n + 1) * 2;
        sb->buf = realloc(sb->buf, sb->cap);
    }
}

static inline void __attribute__((format(printf, 2, 3)))
r300_sb_printf(R300Sb *sb, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    r300_sb_vprintf(sb, fmt, ap);
    va_end(ap);
}

/* Hand ownership of the buffer to the caller. */
static inline char *r300_sb_steal(R300Sb *sb)
{
    char *s = sb->buf;
    sb->buf = NULL;
    sb->len = sb->cap = 0;
    return s;
}

#endif
