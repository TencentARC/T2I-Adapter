# Demos

## Multi-adapters
<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/220939329-379f88b7-444f-4a3a-9de0-8f90605d1d34.png" height=365>
</p>

<div align="center">

*T2I adapters naturally support using multiple adapters together.*

</div><br />
We now only provide the testing script usage for this example:

>python test_composable_adapters.py --prompt "1gril, computer desk, best quality, extremely detailed" --neg_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality" --depth_cond_path examples/depth/desk_depth.png --depth_cond_weight 1.0 --depth_ckpt models/t2iadapter_depth_sd14v1.pth --depth_type_in depth --pose_cond_path examples/keypose/person_keypose.png --pose_cond_weight 1.5 --ckpt models/anything-v4.0-pruned.ckpt --n_sample 4 --max_resolution 524288

[Image source](https://twitter.com/toyxyz3/status/1628375164781211648)
