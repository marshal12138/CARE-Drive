<p align="center">
  <img src="./care-drive-logo.png" alt="CARE-Drive and Chery Next AI">
</p>

<h1 align="center">CARE-Drive: Coupled Alternating Refinement for Ego-trajectories</h1>

<p align="center">
  <a href="https://github.com/chery-nextai">Xiao Ma</a><sup>*</sup>, <a href="https://github.com/chery-nextai">Yijun Chen</a><sup>*</sup>, <a href="https://github.com/chery-nextai">Zhiquan Hu</a>, <a href="https://github.com/chery-nextai">Jicheng Zhao</a>, <a href="https://github.com/chery-nextai">Yuming Li</a><sup>†</sup>, <a href="https://github.com/chery-nextai">Chuang Dai</a>
</p>

<p align="center">NEXTAI Research Institute, Chery Group.</p>

<p align="center"><sub>* Equal contribution. † Corresponding author.</sub></p>

<p align="center">
  <img src="./care-drive-framework.jpg" alt="CARE-Drive framework: predictive backbone, position and heading experts, and alternating expert training">
</p>

CARE-Drive refines ego trajectories for autonomous driving using coupled position and heading experts trained with alternating reinforcement fine-tuning.

## Qualitative demonstrations

### Left turn around a stopped car

At 3.8 s in the NAVSIM replay, the backbone overlaps the stopped car. CARE-Drive clears it by 0.46 m.

![Side-by-side replay of the backbone and CARE-Drive turning left past a stopped car](./01-left-turn-with-stopped-car.gif)

The three panels below show the scored replay and planned pose for the same scene, from the backbone through position and heading correction.

![Three-stage comparison of the backbone, XY correction, and XY plus heading correction at 3.8 seconds](./care-drive-correction-triptych-stopped-car.png)

### Right turn at the drivable-area boundary

At 3.1 s, one corner of the backbone vehicle is 7.4 cm outside the drivable area. The corresponding CARE-Drive corner is 69.5 cm inside.

![Side-by-side right-turn replay with a close-up of the drivable-area boundary](./02-right-turn.gif)

<details>
<summary>More examples</summary>

### Low-speed stop

Both plans preserve the low-speed stopping behavior and score 1.000 PDMS in this scene.

![Low-speed stopping replay where both trajectories are correct](./03-low-speed.gif)

</details>
