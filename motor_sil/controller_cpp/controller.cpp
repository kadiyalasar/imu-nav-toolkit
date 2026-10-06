// Motor controller in C++ (position or velocity) -- same protocol and same PI as controller_py.py.
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

// Same controller as controller_py.py: sign-magnitude PI, Arduino-style inputs and outputs.
struct Command { int pwm, in1, in2; };

class ArduinoController {
 public:
  ArduinoController(bool position_mode, double kp, double ki, double dt, double counts_per_rev,
                    double deadband, bool anti_windup)
      : pos_(position_mode), kp_(kp), ki_(ki), dt_(dt),
        rad_per_count_(2 * kPi / counts_per_rev), deadband_(deadband), aw_(anti_windup),
        full_scale_(position_mode ? 2 * kPi : 25.0 * 2 * kPi / 60) {}

  Command update(long counts, int adc) {
    double ref = adc / 1023.0 * full_scale_;
    double measured;
    if (pos_) {
      measured = counts * rad_per_count_;
    } else {
      long prev = has_prev_ ? prev_counts_ : counts;
      measured = (counts - prev) * rad_per_count_ / dt_;
    }
    prev_counts_ = counts;
    has_prev_ = true;
    double error = ref - measured;
    double u = kp_ * error + ki_ * integral_;
    bool saturated = std::fabs(u) >= 1.0;
    if (!(aw_ && saturated)) integral_ += error * dt_;
    int pwm = static_cast<int>(std::lround(std::fabs(u) * 255));
    if (pwm > 255) pwm = 255;
    if (pos_) {
      if (u > deadband_) return {pwm, 1, 0};
      if (u < -deadband_) return {pwm, 0, 1};
      return {0, 0, 0};
    }
    return u > 0 ? Command{pwm, 1, 0} : Command{pwm, 0, 0};
  }

 private:
  bool pos_;
  double kp_, ki_, dt_, rad_per_count_, deadband_;
  bool aw_;
  double full_scale_;
  double integral_ = 0.0;
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
  int port = -1, anti_windup = 1;
  bool position_mode = true, gains_given = false;
  double kp = 0, ki = 0, counts_per_rev = 8256.0, dt = 0.01, deadband = 0.0, gain_scale = 1.0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) position_mode = std::strcmp(argv[++i], "velocity") != 0;
    else if (!std::strcmp(argv[i], "--gains") && i + 2 < argc) {
      kp = std::atof(argv[++i]); ki = std::atof(argv[++i]); gains_given = true;
    } else if (!std::strcmp(argv[i], "--counts-per-rev") && i + 1 < argc) counts_per_rev = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--dt") && i + 1 < argc) dt = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--deadband") && i + 1 < argc) deadband = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--anti-windup") && i + 1 < argc) anti_windup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gain-scale") && i + 1 < argc) gain_scale = std::atof(argv[++i]);
  }
  if (port < 0) { std::fprintf(stderr, "usage: controller --port N [--mode position|velocity]\n"); return 2; }
  if (!gains_given) {             // same defaults as controller_py.py
    kp = position_mode ? 8.0 : 0.6;
    ki = position_mode ? 1.0 : 10.0;
  }
  ArduinoController ctrl(position_mode, kp * gain_scale, ki * gain_scale, dt, counts_per_rev,
                         deadband, anti_windup != 0);

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
    double t;
    int adc;
    in >> step >> t >> counts >> adc;
    Command c = ctrl.update(counts, adc);
    std::snprintf(out, sizeof(out), "CMD %ld %d %d %d\n", step, c.pwm, c.in1, c.in2);
    if (!send_all(s, out)) break;
  }
  close_sock(s);
#ifdef _WIN32
  WSACleanup();
#endif
  return 0;
}
