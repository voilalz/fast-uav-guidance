#include "nav/protocol.h"

#include <string.h>

uint16_t nav_crc16(const uint8_t* data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

size_t nav_encode(uint8_t msg_id, uint16_t seq, const void* payload, uint8_t len, uint8_t* out, size_t out_cap) {
  const size_t total = NAV_HEADER_LEN + (size_t)len + NAV_CRC_LEN;
  if (out_cap < total) return 0;
  out[0] = NAV_SYNC0;
  out[1] = NAV_SYNC1;
  out[2] = msg_id;
  out[3] = len;
  out[4] = (uint8_t)(seq & 0xFF);
  out[5] = (uint8_t)(seq >> 8);
  if (len) memcpy(out + NAV_HEADER_LEN, payload, len);
  const uint16_t crc = nav_crc16(out + 2, 4u + len);
  out[NAV_HEADER_LEN + len] = (uint8_t)(crc & 0xFF);
  out[NAV_HEADER_LEN + len + 1] = (uint8_t)(crc >> 8);
  return total;
}

void nav_decoder_init(nav_decoder_t* d) {
  d->pos = 0;
  d->need = NAV_HEADER_LEN;
  d->crc_errors = 0;
}

int nav_decoder_push(nav_decoder_t* d, uint8_t byte) {
  if (d->pos == 0 && byte != NAV_SYNC0) return 0;
  if (d->pos == 1 && byte != NAV_SYNC1) {
    d->pos = (byte == NAV_SYNC0) ? 1 : 0;
    return 0;
  }
  d->buf[d->pos++] = byte;
  if (d->pos == NAV_HEADER_LEN) d->need = NAV_HEADER_LEN + d->buf[3] + NAV_CRC_LEN;
  if (d->pos < d->need) return 0;

  const size_t len = d->buf[3];
  const uint16_t crc = (uint16_t)(d->buf[NAV_HEADER_LEN + len] | (d->buf[NAV_HEADER_LEN + len + 1] << 8));
  d->pos = 0;
  d->need = NAV_HEADER_LEN;
  if (crc != nav_crc16(d->buf + 2, 4u + len)) {
    d->crc_errors++;
    return 0;
  }
  return 1;
}
