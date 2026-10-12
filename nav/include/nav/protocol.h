/*
 * 导航 MCU <-> 视觉计算机（Jetson）<-> 飞控 之间的二进制消息协议。
 * C99，小端，1 字节对齐。帧格式：
 *   [sync0=0xA5][sync1=0x5A][msg_id][len][seq u16][payload len 字节][crc16 u16]
 * crc16 为 CRC-16/CCITT-FALSE，覆盖 msg_id..payload。
 * 物理链路：MCU<->Jetson 用 100 Mbps 以太网 UDP（每个 UDP 包一帧），
 *           MCU->飞控 用 UART 921600 bps。
 */
#ifndef NAV_PROTOCOL_H
#define NAV_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NAV_SYNC0 0xA5u
#define NAV_SYNC1 0x5Au
#define NAV_HEADER_LEN 6u
#define NAV_CRC_LEN 2u
#define NAV_MAX_PAYLOAD 255u
#define NAV_MAX_FRAME (NAV_HEADER_LEN + NAV_MAX_PAYLOAD + NAV_CRC_LEN)

enum nav_msg_id {
  NAV_MSG_NAV_OUTPUT = 0x01,   /* MCU -> 飞控/Jetson，30 Hz */
  NAV_MSG_TIME_SYNC = 0x02,    /* MCU -> Jetson，1 Hz，配合 PPS 线 */
  NAV_MSG_CAM_TRIGGER = 0x03,  /* MCU -> Jetson，每次触发相机 */
  NAV_MSG_NAV_PRIOR = 0x04,    /* MCU -> Jetson，10 Hz，匹配用先验位置/姿态/协方差 */
  NAV_MSG_VO_VELOCITY = 0x10,  /* Jetson -> MCU，30 Hz */
  NAV_MSG_POSITION_FIX = 0x11, /* Jetson -> MCU，景象/地形匹配结果 */
  NAV_MSG_HEALTH = 0x20,       /* 双向，1 Hz */
};

#pragma pack(push, 1)

typedef struct {
  uint64_t t_us;
  double lat_rad, lon_rad;
  float alt_m;
  float vel_ned[3];
  float euler_rad[3];
  float gyro_rad_s[3];
  float sigma_ne_m, sigma_d_m;
  uint8_t mode;
  uint8_t health;
} nav_output_t;

typedef struct {
  uint64_t t_pps_us; /* 最近一次 PPS 上升沿的 MCU 时间 */
  uint32_t pps_count;
} nav_time_sync_t;

typedef struct {
  uint64_t t_mid_exposure_us; /* 曝光中点时间 */
  uint32_t frame_id;
  uint8_t camera; /* 0 可见光，1 红外 */
  uint16_t exposure_us;
} nav_cam_trigger_t;

typedef struct {
  uint64_t t_us;
  double lat_rad, lon_rad;
  float alt_m;
  float euler_rad[3];
  float sigma_ne_m;
  float agl_m;
} nav_prior_t;

typedef struct {
  uint64_t t_us; /* 对应帧的曝光中点 */
  uint32_t frame_id;
  float vb[3];   /* 机体系速度 m/s */
  float sigma;   /* m/s */
  uint16_t inliers;
  uint8_t quality; /* 0-100 */
} nav_vo_velocity_t;

typedef struct {
  uint64_t t_us; /* 曝光中点 */
  uint32_t frame_id;
  double lat_rad, lon_rad;
  float alt_m;
  float sigma_ne_m;
  float sigma_d_m; /* <0 表示无高度 */
  float yaw_rad;   /* 景象匹配航向，NaN 表示无 */
  uint8_t source;  /* 1 景象 2 三维地形 3 TERCOM */
  uint8_t quality;
  uint16_t inliers;
} nav_position_fix_t;

typedef struct {
  uint64_t t_us;
  uint8_t node; /* 0 MCU 1 Jetson */
  uint8_t cpu_load;
  int8_t temp_c;
  uint8_t flags;
  uint32_t err_count;
} nav_health_t;

#pragma pack(pop)

uint16_t nav_crc16(const uint8_t* data, size_t len);

/* 编码一帧，返回帧长度；缓冲区不足返回 0 */
size_t nav_encode(uint8_t msg_id, uint16_t seq, const void* payload, uint8_t len, uint8_t* out, size_t out_cap);

/* 流式解码器：逐字节喂入，收到完整且校验正确的帧时返回 1 */
typedef struct {
  uint8_t buf[NAV_MAX_FRAME];
  size_t pos;
  size_t need;
  uint32_t crc_errors;
} nav_decoder_t;

void nav_decoder_init(nav_decoder_t* d);
int nav_decoder_push(nav_decoder_t* d, uint8_t byte);
static inline uint8_t nav_frame_id(const nav_decoder_t* d) { return d->buf[2]; }
static inline uint8_t nav_frame_len(const nav_decoder_t* d) { return d->buf[3]; }
static inline const uint8_t* nav_frame_payload(const nav_decoder_t* d) { return d->buf + NAV_HEADER_LEN; }

#ifdef __cplusplus
}
#endif

#endif /* NAV_PROTOCOL_H */
