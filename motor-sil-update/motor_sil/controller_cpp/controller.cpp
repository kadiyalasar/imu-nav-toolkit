// Motor position controller in C++ -- same protocol and same PID as controller_py.py.
// This plays the role of the "production firmware" in software-in-the-loop.
//
// Build (Windows, MinGW): g++ -O2 -std=c++17 controller.cpp -o controller.exe -lws2_32
// Build (Linux/macOS):    g++ -O2 -std=c++17 controller.cpp -o controller
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  using sock_t = SOCKET;
  static void close_sock(sock_t s) { closesocket(s); }
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <unistd.h>
  using sock_t = int;
  static void close_sock(sock_t s) { close(s); }
#endif

constexpr double kPi = 3.141592653589793;

// PID on output angle: derivative on measurement, filtered speed estimate, anti-windup.
class Pid {
 public:
  Pid(double kp, double ki, double kd, double dt, double counts_per_rev, double d_filter_hz = 50.0)
      : kp_(kp), ki_(ki), kd_(kd), dt_(dt),
        rad_per_count_(2 * kPi / counts_per_rev),
        alpha_(dt / (dt + 1 / (2 * kPi * d_filter_hz))) {}

  double update(long counts, double target) {
    double angle = counts * rad_per_count_;
    if (has_prev_) {
      double raw_speed = (counts - prev_counts_) * rad_per_count_ / dt_;
      speed_ += alpha_ * (raw_speed - speed_);
    }
    prev_counts_ = counts;
    has_prev_ = true;
    double error = target - angle;
    double u = kp_ * error + ki_ * integral_ - kd_ * speed_;
    double u_sat = u > 1.0 ? 1.0 : (u < -1.0 ? -1.0 : u);
    if (u == u_sat) integral_ += error * dt_;   // anti-windup
    return u_sat;
  }

 private:
  double kp_, ki_, kd_, dt_, rad_per_count_, alpha_;
  double integral_ = 0.0, speed_ = 0.0;
  long prev_counts_ = 0;
  bool has_prev_ = false;
};

class LineReader {
 public:
  explicit LineReader(sock_t s) : sock_(s) {}
  bool next(std::string& line) {
    for (;;) {
      auto pos = buf_.find('\n');
      if (pos != std::string::npos) {
        line = buf_.substr(0, pos);
        buf_.erase(0, pos + 1);
        return true;
      }
      char chunk[512];
      int n = recv(sock_, chunk, sizeof(chunk), 0);
      if (n <= 0) return false;
      buf_.append(chunk, n);
    }
  }
 private:
  sock_t sock_;
  std::string buf_;
};

static bool send_all(sock_t s, const std::string& msg) {
  size_t sent = 0;
  while (sent < msg.size()) {
    int n = send(s, msg.data() + sent, static_cast<int>(msg.size() - sent), 0);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

int main(int argc, char** argv) {
  int port = -1;
  double kp = 12.0, ki = 60.0, kd = 0.2, counts_per_rev = 1440.0, dt = 1e-3, gain_scale = 1.0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gains") && i + 3 < argc) {
      kp = std::atof(argv[++i]); ki = std::atof(argv[++i]); kd = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--counts-per-rev") && i + 1 < argc) counts_per_rev = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--dt") && i + 1 < argc) dt = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--gain-scale") && i + 1 < argc) gain_scale = std::atof(argv[++i]);
  }
  if (port < 0) { std::fprintf(stderr, "usage: controller --port N\n"); return 2; }
  Pid pid(kp * gain_scale, ki * gain_scale, kd * gain_scale, dt, counts_per_rev);

#ifdef _WIN32
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return 1;
#endif
  sock_t s = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<unsigned short>(port));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::fprintf(stderr, "connect failed\n");
    return 1;
  }
  int one = 1;  // send small messages immediately (disable Nagle)
  setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));

  send_all(s, "HELLO cpp\n");
  LineReader reader(s);
  std::string line;
  char out[128];
  while (reader.next(line)) {
    std::istringstream in(line);
    std::string tag;
    in >> tag;
    if (tag == "END") break;
    if (tag != "STATE") continue;
    long step, counts;
    double t, target;
    in >> step >> t >> counts >> target;
    double duty = pid.update(counts, target);
    std::snprintf(out, sizeof(out), "CMD %ld %.17g\n", step, duty);
    if (!send_all(s, out)) break;
  }
  close_sock(s);
#ifdef _WIN32
  WSACleanup();
#endif
  return 0;
}
