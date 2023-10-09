# Prioritizing Emergency Evacuations using POMDPs: A Case Study of the Afghanistan Evacuation
_Prioritizing emergency evacuations under compounding levels of uncertainty_, 2022 IEEE GHTC submission.

<p align="center">
  <img src="./media/trajectory.svg">
</p>

## Installation

From a termal, run:
```bash
julia install.jl
```

## Supplementary Appendix: Population Estimates
Our estimation of the number of Afghans to be evacuated came from data calculated by [the Association of Wartime Allies](https://drive.google.com/file/d/1NXtSlu0_A38Vj9d7w4hcLizv4f0kFVOg/view), who calculated numbers by collating publicly available data published by the Department of Defense, Department of State, and other non-governmental organizations and research organizations. Its estimates were developed on August 25, 2021 as an estimate of how many people remained at that time. We used the numbers reflected in its data, but added the number of people who they estimated had already been evacuated. For example, if at the time of their report 118,000 SIV applicants and family members were on the ground in Afghanistan, with 5000 already evacuated, the adjusted number at the beginning of the evacuation would be 123,000 SIVs with their families. For categories with low and high estimates, we took the average of the estimates.
