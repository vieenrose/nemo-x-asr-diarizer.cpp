#!/bin/bash
# Arm the CPU: wake + volume keyevents, then keep a background wake-stream alive so the scheduler holds the
# prime core at max frequency. Mirrors arm_start() in measure.sh; without it a phone drifts to ~1.3 GHz and
# every number is 20-60% slow while looking completely reasonable. See also .auto/witness.sh.
D=${D:-/data/local/tmp/nemo_x}
adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true
for _i in 1 2 3; do adb shell "input keyevent KEYCODE_VOLUME_DOWN" >/dev/null 2>&1 || true; adb shell "input keyevent KEYCODE_VOLUME_UP" >/dev/null 2>&1 || true; sleep 1; done
# The sentinel must be cleared here, not only in disarm.sh: disarm touches it to stop the previous stream,
# so a stale file makes the new background loop exit on its first test and the device drifts to 1.3 GHz
# while everything still prints plausible numbers. This bug cost a whole probe batch.
rm -f /tmp/ar_arm_done /tmp/ar_arm_stream
( while [ ! -e /tmp/ar_arm_done ]; do adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1 || true; sleep 3; done ) >/dev/null 2>&1 &
echo $! > /tmp/ar_arm_pid
sleep 1; echo armed
