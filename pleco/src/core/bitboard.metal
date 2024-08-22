#include <metal_stdlib>
using namespace metal;

kernel void bitboard_operations(device const uint64_t* bitboards [[buffer(0)]],
                                device uint8_t* results [[buffer(1)]],
                                uint index [[thread_position_in_grid]],
                                constant uint& buffer_size [[buffer(2)]])
{
    if (index >= buffer_size) {
        return;
    }
    
    uint64_t bb = bitboards[index];
    results[index] = popcount(bb);
}