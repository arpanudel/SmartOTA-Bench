# Bounded Profile Notes

The final deterministic benchmark used all enabled reproducible artifact pairs. Block sizes 16384 and 65536 were run for every enabled pair. Block size 4096 was run only for the small Autoware-style perception module and YOLO-small container fixture. The 4096-byte dimension was skipped for Alpine compressed, Alpine normalized/rootfs, and YOLO multiblock because a rootfs probe took about 63 seconds for only eight policy rows under one deployment and one interruption setting, making the full large-artifact 4096 matrix disproportionate for this final deterministic run.

Seeds: deterministic seed 1 only. Learned methods and the 10-vehicle deployment MVP were not run.
