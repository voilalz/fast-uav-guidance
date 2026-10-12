# fast-uav-guidance

高速（约 600 km/h）小型固定翼无人机在 GNSS 拒止条件下的自主导航方案研究。

- [总体方案](docs/navigation-design.md)：架构、景象匹配/地形匹配、ESKF 融合、误差预算、仿真与验证计划
- [硬件方案](docs/hardware-design.md)：传感器、处理器、飞控、执行机构选型与电气连接
- [软件方案](docs/software-design.md)：MCU 固件与 Jetson 视觉服务架构、滤波器设计、协议、验证结果
- [导航代码](nav/)：嵌入式 C++17 / C99 导航核心库、闭环仿真与单元测试
- [误差预算脚本](tools/error_budget.py)：`python3 tools/error_budget.py`
- [实验记录](docs/experiments.md)：门限锁死 Monte-Carlo 对比实验、真实影像上的景象匹配与 VO 评估
- [视觉离线管线](vision/)：Python 实现的 VO 与景象匹配（真实多时相卫星影像 + DEM 渲染）
- [Monte-Carlo 实验脚本](tools/experiments.py)

## 快速开始

```bash
cd nav
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/nav_sim 1200 1
./build/nav_mc runs=100 out=mc profile=realistic recovery=1
```
