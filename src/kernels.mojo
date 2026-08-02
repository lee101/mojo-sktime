"""Time-series kernels exposed through a non-parametric C ABI."""

from std.math import sqrt

comptime PtrF = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime PtrI = UnsafePointer[Int64, AnyOrigin[mut=True]]


def pf(address: Int) -> PtrF:
    return PtrF(unsafe_from_address=address)


def pi(address: Int) -> PtrI:
    return PtrI(unsafe_from_address=address)


def paa_frame_value(x: PtrF, length: Int, frames: Int, frame: Int) -> Float64:
    var begin = frame * length
    var end = begin + length
    var first_source = begin // frames
    var last_source = (end - 1) // frames
    var total = 0.0
    for source in range(first_source, last_source + 1):
        var left = source * frames
        if left < begin:
            left = begin
        var right = (source + 1) * frames
        if right > end:
            right = end
        total += Float64(right - left) * x[source]
    return total / Float64(length)


@export("mts_paa_frames")
def mts_paa_frames(x_addr: Int, result_addr: Int, length: Int, frames: Int) abi("C"):
    var x = pf(x_addr)
    var result = pf(result_addr)
    for frame in range(frames):
        result[frame] = paa_frame_value(x, length, frames, frame)


@export("mts_paa_frame_size")
def mts_paa_frame_size(x_addr: Int, result_addr: Int, length: Int, frame_size: Int) abi("C"):
    var x = pf(x_addr)
    var result = pf(result_addr)
    var frames = (length + frame_size - 1) // frame_size
    for frame in range(frames):
        var begin = frame * frame_size
        var end = begin + frame_size
        if end > length:
            end = length
        var total = 0.0
        for j in range(begin, end):
            total += x[j]
        result[frame] = total / Float64(end - begin)


@export("mts_sax_frames")
def mts_sax_frames(
    x_addr: Int, result_addr: Int, breakpoints_addr: Int, length: Int, frames: Int,
    breakpoint_count: Int,
) abi("C"):
    var x = pf(x_addr)
    var result = pi(result_addr)
    var breakpoints = pf(breakpoints_addr)
    var mean = 0.0
    for i in range(length):
        mean += x[i]
    mean /= Float64(length)
    var variance = 0.0
    for i in range(length):
        var delta = x[i] - mean
        variance += delta * delta
    var scale = sqrt(variance / Float64(length))
    for frame in range(frames):
        var begin = frame * length
        var end = begin + length
        var first_source = begin // frames
        var last_source = (end - 1) // frames
        var total = 0.0
        for source in range(first_source, last_source + 1):
            var left = source * frames
            if left < begin:
                left = begin
            var right = (source + 1) * frames
            if right > end:
                right = end
            total += Float64(right - left) * ((x[source] - mean) / scale)
        var value = total / Float64(length)
        var symbol = 0
        while symbol < breakpoint_count and value >= breakpoints[symbol]:
            symbol += 1
        result[frame] = Int64(symbol)


@export("mts_sax_frame_size")
def mts_sax_frame_size(
    x_addr: Int, result_addr: Int, breakpoints_addr: Int, length: Int, frame_size: Int,
    breakpoint_count: Int,
) abi("C"):
    var x = pf(x_addr)
    var result = pi(result_addr)
    var breakpoints = pf(breakpoints_addr)
    var mean = 0.0
    for i in range(length):
        mean += x[i]
    mean /= Float64(length)
    var variance = 0.0
    for i in range(length):
        var delta = x[i] - mean
        variance += delta * delta
    var scale = sqrt(variance / Float64(length))
    var frames = (length + frame_size - 1) // frame_size
    for frame in range(frames):
        var begin = frame * frame_size
        var end = begin + frame_size
        if end > length:
            end = length
        var total = 0.0
        for j in range(begin, end):
            total += (x[j] - mean) / scale
        var value = total / Float64(end - begin)
        var symbol = 0
        while symbol < breakpoint_count and value >= breakpoints[symbol]:
            symbol += 1
        result[frame] = Int64(symbol)


@export("mts_pairwise_euclidean")
def mts_pairwise_euclidean(
    train_addr: Int, query_addr: Int, result_addr: Int, train_count: Int, query_count: Int,
    channels: Int, timepoints: Int,
) abi("C"):
    var train = pf(train_addr)
    var query = pf(query_addr)
    var result = pf(result_addr)
    var width = channels * timepoints
    for qi in range(query_count):
        for ti in range(train_count):
            var total = 0.0
            for j in range(width):
                var delta = query[qi * width + j] - train[ti * width + j]
                total += delta * delta
            result[qi * train_count + ti] = sqrt(total)


@export("mts_pairwise_dtw")
def mts_pairwise_dtw(
    train_addr: Int, query_addr: Int, result_addr: Int, scratch_addr: Int,
    train_count: Int, query_count: Int, channels: Int, timepoints: Int,
) abi("C"):
    var train = pf(train_addr)
    var query = pf(query_addr)
    var result = pf(result_addr)
    var scratch = pf(scratch_addr)
    var inf = 1.7976931348623157e308
    for qi in range(query_count):
        for ti in range(train_count):
            for j in range(timepoints + 1):
                scratch[j] = inf
            scratch[0] = 0.0
            for i in range(1, timepoints + 1):
                scratch[timepoints + i] = inf
                for j in range(1, timepoints + 1):
                    var cost = 0.0
                    for channel in range(channels):
                        var delta = query[(qi * channels + channel) * timepoints + i - 1] - train[(ti * channels + channel) * timepoints + j - 1]
                        cost += delta * delta
                    var best = scratch[j]
                    if scratch[timepoints + j - 1] < best:
                        best = scratch[timepoints + j - 1]
                    if scratch[j - 1] < best:
                        best = scratch[j - 1]
                    scratch[timepoints + j] = cost + best
                for j in range(timepoints + 1):
                    scratch[j] = scratch[timepoints + j]
            result[qi * train_count + ti] = scratch[timepoints]
