#!/bin/bash
[ -f /tmp/ar_arm_pid ] && kill "$(cat /tmp/ar_arm_pid)" 2>/dev/null; rm -f /tmp/ar_arm_pid /tmp/ar_arm_done
touch /tmp/ar_arm_done 2>/dev/null; adb shell "input keyevent KEYCODE_SLEEP" >/dev/null 2>&1 || true; echo disarmed
