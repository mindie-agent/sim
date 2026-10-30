/*
[Role] Fast independent max-plus recurrence for primitive parameter identification.
[Invariants] No model or device runtime. Each queue entry releases on response;
  requests to the same address granule serialize. Durations use nanoseconds.
[Timeouts & Locks] Called on a single <=64-row compiler tile; at most 64 entries.
[Side-Effect Policy] None. Production prediction uses the shared Python Engine.
*/
#include <algorithm>
#include <array>
#include <map>

extern "C" double tile(int rows, int cols, int stride, int offset, int granule,
                       int capacity, double response, double issue, double row_control) {
    std::array<double, 64> pending{};
    std::map<int, double> last;
    double clock = 0;
    int used = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto slot = std::min_element(pending.begin(), pending.begin() + used);
            if (used == capacity) clock = std::max(clock, *slot);
            else slot = pending.begin() + used++;
            clock += issue;
            const int address_granule = (row * stride + col + offset) * 2 / granule;
            *slot = std::max(clock, last[address_granule]) + response;
            last[address_granule] = *slot;
        }
        clock += row_control;
    }
    return std::max(clock, *std::max_element(pending.begin(), pending.begin() + used));
}
