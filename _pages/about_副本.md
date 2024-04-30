---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Algorithmic research revolves around multimodal interaction and modeling, with internship tasks involving LLM and image generation. Currently seeking positions related to multimodal algorithms and AIGC.

I will obtain master's degree from the University of Science and Technology of China, supervised by Associate Professor Yu Jun, with corporate mentors Peng Chang, the head of the multimodal group at the Silicon Valley Research Institute of Ping An Technology in the United States, and Iek-Heng Chu. I graduated from Guangzhou University with a bachelor's degree, supervised by Professor Jin Li, the executive dean of the Institute of Artificial Intelligence, and Associate Professor Xianmin Wang. Currently, I have contributed to the publication of more than 10 articles. <a href='https://scholar.google.com/citations?user=MmZ_y1QAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>.

During my undergraduate years, I frequently participated in algorithm competitions and wrote [blogs](https://blog.csdn.net/weixin_43999137?type=blog). I was a member of the Atmosphere team and served as a member of the Alibaba Security Student Expert Group. I was ranked in the [Top 10 of the Alibaba Security Challenger Program](https://s.alibaba.com/challenge?spm=a2c22.12281976.0.0.46db2a69WaN1Te).

My research interests include:
- Multimodal interaction and modeling (CV/NLP)
- AIGC
- Fine-grained image recognition
- Robust machine learning

My business directions include:
- Large language models
- Exploratory data analysis (EDA)
- Data mining
- Style transfer (AE, GAN, Diffusion)
- Object detection

<!-- <span class='anchor' id='-news'></span>

# 🔥 News -->


<span class='anchor' id='-Publications'></span>

# 📝 Published Papers



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">IJCAI 2024(CCF-A)</div><img src='images/CEAM.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Dialogue Cross-Enhanced Central Engagement Attention Model for Real-Time Engagement Estimation]() \\
Jun Yu, **Keda Lu**, Ji Zhao et al. (primary student author)

1. For real-time engagement estimation, deep exploration is conducted on sliding window and model perspectives.
2. To solve the problem of repetitive inference in sliding windows, a center-based sliding window is proposed, improving inference efficiency by 100%.
3. A core engagement attention model based on self-attention mechanism is proposed, surpassing the previous SOTA BiLSTM model, with inference efficiency improved by 300%.
4. Based on cross attention, a cross-enhanced module is proposed and seamlessly integrated with the core engagement attention model, enhancing real-time engagement estimation to a new SOTA level.

<!-- [**Project**](https://portaspeech.github.io/) \| [![](https://img.shields.io/github/stars/NATSpeech/NATSpeech?style=social&label=Code+Stars)](https://github.com/NATSpeech/NATSpeech) \| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/NATSpeech/PortaSpeech) -->
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024(CCF-A) workshop</div><img src='images/mvav.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[MvAV-pix2pixHD: Multi-view Aerial View Image Translation]() \\
Jun Yu, **Keda Lu**, Shenshen Du et al. (primary student author)

1. Time-priority sampling and random sampling methods are designed for multi-view image translation tasks.
2. MvAV-pix2pixHD is proposed for multi-view image translation, using three powerful losses.
3. The method of this paper is applied to 2 multi-view image translation tasks in the MAVIC-T competition, winning 1 champion and 1 runner-up

<!-- [**Project**](https://portaspeech.github.io/) \| [![](https://img.shields.io/github/stars/NATSpeech/NATSpeech?style=social&label=Code+Stars)](https://github.com/NATSpeech/NATSpeech) \| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/NATSpeech/PortaSpeech) -->
</div>
</div>




- `ACM-MM 2023` [Answer-Based Entity Extraction and Alignment for Visual Text Question Answering](https://dl.acm.org/doi/abs/10.1145/3581783.3612850) Jun Yu, Mohan Jing, Weihao Liu, Tongxu Luo, Bingyuan Zhang, **Keda Lu** et al.


- `CLEF 2022` [Bag of Tricks and a Strong Baseline for FGVC.](https://ceur-ws.org/Vol-3180/paper-182.pdf) Jun Yu, Hao Chang, **Keda Lu** et al.


- `CLEF 2022` [Efficient Model Integration for Snake Classification](https://ceur-ws.org/Vol-3180/paper-181.pdf) Jun Yu, Hao Chang, Zhongpeng Cai, Guochen Xie, Liwen Zhang, **Keda Lu** et al.


- `CVPR 2022 workshop` [Pseudo-label generation and various data augmentation for semi-supervised hyperspectral object detection](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html) Jun Yu, Liwen Zhang, Shenshen Du, Hao Chang, **Keda Lu** et al.


- `AAAI 2022 workshop` [Mining limited data for more robust and generalized ML models](https://alisec-competition.oss-cn-shanghai.aliyuncs.com/competition_papers/20211201/rank10.pdf), Jun Yu, Hao Chang, **Keda Lu** et al.


- `International Journal of Machine Learning and Cybernetics` [Generating transferable adversarial examples based on perceptually-aligned perturbation](https://link.springer.com/article/10.1007/s13042-020-01240-1), Hongqiao Chen, **Keda Lu**, Xianmin Wang et al.



<span class='anchor' id='-projects'></span>

# 💻 业务项目


<span class='anchor' id='-competitions'></span>

# 🏅 参与竞赛

### 研究生阶段（主要参与）
---

- *2024.03* [CVPR 2024: Multi-modal Aerial View Image Challenge - T (Translation)](https://codalab.lisn.upsaclay.fr/competitions/17224) (Top3奖金池2500$, 个人solo, **亚军**🥈) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/17224#results) [[论文]]


- *2023.10* [ICCV 2023: Out Of Distribution Generalization: Object Classification track](https://codalab.lisn.upsaclay.fr/competitions/14068#results) (个人solo, **冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/14068#results) [[论文在投]]


- *2023.10* [ICCV 2023: Out Of Distribution Generalization: Pose Estimation track](https://codalab.lisn.upsaclay.fr/competitions/14074#learn_the_details) (个人solo, **冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/14074#results) [[技术报告]](https://www.ood-cv.org/reports/pose/ImageNet1k-1st.pdf)


- *2023.07* [ACM-MM 2023: Grand challenge, Engagement Estimation](https://multimediate-challenge.org/) (个人solo, **冠军**🏆) [[Leaderboard]](https://multimediate-challenge.org/leaderboards/leaderboard_engagement/) [[论文]](https://dl.acm.org/doi/abs/10.1145/3581783.3612852) [[New]](https://cloud.tencent.com/developer/news/1167803)


- *2022.10* [ECCV 2022: Out Of Distribution Generalization Track-1: Object Classification](https://www.ood-cv.org/challenge.html) (Top3奖金池3300$, 个人solo, **亚军**🥈) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/6781#results) [[Code]](https://github.com/wujiekd/ECCV2022-OOD-CV-Challenge-Classification-Track-2nd-USTC-IAT-United) 


- *2022.10* [ECCV 2022: Out Of Distribution Generalization Track-2: Object Detection](https://www.ood-cv.org/challenge.html) (Top3奖金池3300$, **亚军**🥈) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/6784#results) [[Code]](https://github.com/wujiekd/ECCV2022-OOD-CV-Challenge-detection-Track-2nd-Place-Program) 


- *2022.05* [CVPR 2022: FGVC9 workshop FungiCLEF2022 challenge](https://sites.google.com/view/fgvc9/competitions/fungiclef2022) (**亚军**🥈) [[Leaderboard]](https://www.kaggle.com/c/fungiclef2022/leaderboard) [[Code]](https://github.com/wujiekd/Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification) [[论文]](https://ceur-ws.org/Vol-3180/paper-182.pdf) 


- *2022.03* [CVPR 2022: Multi-modal Aerial View Object Classification - Track 2 (SAR+EO)](https://codalab.lisn.upsaclay.fr/competitions/1392) (Top3奖金池6000$, **冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/1392#results) [[技术报告]](https://arxiv.org/abs/2205.01920) [[New]](https://zhuanlan.zhihu.com/p/493603389)


- *2022.03* [CVPR 2022: Multi-modal Aerial View Object Classification - Track 1 (SAR)](https://codalab.lisn.upsaclay.fr/competitions/1388) (Top3奖金池6000$, **冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/1388#results) [[技术报告]](https://arxiv.org/abs/2205.01920) [[New]](https://zhuanlan.zhihu.com/p/493603389)


### 研究生阶段（协助参与）
<!-- MM jmh 季军 iccv zj 冠军 cvpr shooc 和 mavic 冠军 ROSE冠军 -->
---
- *2023.12* [ICCV 2023: WECIA - Caption Generation Challenge](https://eval.ai/web/challenges/challenge-page/2104/overview) (**冠军**🏆) [[Leaderboard]](https://eval.ai/web/challenges/challenge-page/2104/leaderboard/5203)


- *2023.07* [ACM-MM 2023: Visual Text Question Answering](https://visual-text-qa.github.io/) (**季军**🥉) [[Leaderboard]](http://vtqa-challenge.fixtankwun.top:20010/) [[论文]](https://dl.acm.org/doi/abs/10.1145/3581783.3612850)


- *2023.03* [CVPR 2023: Multi-modal Aerial View Imagery Challenges - Translation](https://codalab.lisn.upsaclay.fr/competitions/9968) (Top3奖金池2250$, **冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/9968#results) [[论文]](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_8) 


- *2022.06* [CVPR 2022: Robustness in Sequential Data challenge](https://codalab.lisn.upsaclay.fr/competitions/2618#learn_the_details) (**冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/2618#results) [[技术报告]](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/USTC_IAT_first_solution_rose_challenge_22.pdf) [[New]](https://nelslip.ustc.edu.cn/2022/0608/c26914a562921/page.htm)


- *2022.03* [CVPR 2022: Semi-Supervised Hyperspectral Object Detection Challenge](https://codalab.lisn.upsaclay.fr/competitions/1752) (**冠军**🏆) [[Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/1752#results) [[论文]](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html)


### 本科生阶段
---

- *2022.08* [中国高校计算机大赛—微信大数据挑战赛](https://algo.weixin.qq.com/) (前十奖金池56万¥, 个人solo, **全国二等奖**, Top30/3000+)  [[Leaderboard]](https://algo.weixin.qq.com/) [[Code]](https://github.com/wujiekd/WeChat-Big-Data-Challenge-2022-National-Second-Prize-Top30) 


- *2022.01* [AAAI 2022：以数据为中心的鲁棒机器学习竞赛](https://advml-workshop.github.io/aaai2022/) (前十奖金池100万¥, 个人solo, 初赛2/3692, **复赛10/3692**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/531939/rankingList) [[Code]](https://github.com/wujiekd/RTA-Iterative-Search-AAAI2022) [[论文]](https://alisec-competition.oss-cn-shanghai.aliyuncs.com/competition_papers/20211201/rank10.pdf) 


- *2021.11* [OPPO安全AI挑战赛-人脸识别攻击](https://security.oppo.com/challenge/home.html) (前十奖金池60万¥, 个人solo, 初赛6/2000+, **复赛12/2000+**) [[Leaderboard]](https://security.oppo.com/challenge/rank.html) [[Code]](https://github.com/wujiekd/Hot-restart-black-box-face-adversarial-attack) 


- *2021.03* [CVPR 2021：对抗机器学习研讨会竞赛, 防御模型的白盒对抗攻击](https://aisecure-workshop.github.io/amlcvpr2021/) (前十奖金池10万$, 个人solo, **排名20/1681**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/531847/rankingList) [[Code]](https://github.com/wujiekd/CVPR2021_ODI_BIM_Attack?spm=a2c22.21852664.0.0.7830775fHm2G8V)  [[Blog]](https://tianchi.aliyun.com/forum/post/208313)

- *2020.10* [伪造图像的对抗攻击竞赛(阿里天池和清华大学联合举办)](https://tianchi.aliyun.com/competition/entrance/531812) (Top 10 prize pool 2 million ¥, **ranking 6/1666**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/531812/rankingList)


- *2020.08* [腾讯广告算法大赛](https://algo.qq.com/) (Top 10 prize pool $100,000, 6th place in preliminary round, **11th place in repechage**, 10,000+ participants) [[Code]](https://github.com/wujiekd/2020-Tencent-advertising-algorithm-contest-rank11) [[Blog]](https://blog.csdn.net/weixin_43999137/article/details/107657517?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171314962516800222817673%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171314962516800222817673&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-107657517-null-null.142^v100^pc_search_result_base3&utm_term=%E8%85%BE%E8%AE%AF%E5%B9%BF%E5%91%8A%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9Bwujiekd&spm=1018.2226.3001.4187)


- *2020.04* [二手车交易价格预测正式赛(阿里天池联合Datawhale举办)](https://tianchi.aliyun.com/competition/entrance/231784) (Individual Solo, Winner,  **排名13/2815**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/231784/rankingList) [[Code]](https://github.com/wujiekd/Predicting-used-car-prices) [[Blog]](https://tianchi.aliyun.com/forum/post/104728) 

- *2020.03* [文本分类对抗攻击竞赛(阿里天池和清华大学联合举办)](https://tianchi.aliyun.com/competition/entrance/231762) (Top 10 prize Pool 68,000¥, **排名4/1666**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/231762/rankingList) [[Code]](https://github.com/wujiekd/NLP_Chinese_adversarial_attack) [[Blog]](https://tianchi.aliyun.com/forum/post/95886) 



- *2019.12* [ImageNet图像分类对抗(阿里天池和清华大学联合举办)](https://tianchi.aliyun.com/competition/entrance/231761) (Top 10 Prize Pool 68,000¥, **排名5/1522**) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/231761/rankingList) [[Blog]](https://tianchi.aliyun.com/forum/post/87389)


- *2019.10* [GeekPwn2019 International Security Geek Contest CAAD CTF Finals](https://geekcon.top/hof/zh/index.html) (Finals prize pool of 100,000¥, **5th place in the Shanghai Finals**, the only undergraduate student team) [[Leaderboard]](https://tianchi.aliyun.com/competition/entrance/231784/rankingList) [[New]](https://www.gzhu.edu.cn/info/1070/3803.htm) 


<span class='anchor' id='-honors'></span>

# 🎖 荣誉奖项
- *2023.11* Huawei Scholarship (Top 30th in the university)
- *2023.10* National Scholarship (Top 1% of graduate students)
- *2022.10* National Scholarship (Top 1% of graduate students)
- *2021.10* National Scholarship (Top 1% of undergraduate students)
- *2020.10* National Scholarships (Top 1% of undergraduate)


<span class='anchor' id='-educations'></span>

# 🎓 教育
- *2022.09 - 2025.07*, <a href="https://www.ustc.edu.cn/"><img class="svg" src="/images/ustclogo.png" width="23pt"></a> University of Science and Technology of China, Computer Technology, Recommended Exemptions, Master's Degree
- *2018.09 - 2022.06*, <a href="https://www.gzhu.edu.cn/"><img class="svg" src="/images/gzhulogo.png" width="20pt"></a> Guangzhou University, Computer Science and Technology (1/591), Bachelor's Degree


<span class='anchor' id='-meetings'></span>

# 🏛️ 学术会议
- *2024.03*, Sense AI Framework Industry Summit (organized by Huawei), invited by Huawei, Beijing.
- *2023.11*, 31st ACM International Conference on Multimedia, Ottawa, Canada.
- *2020.12*, The 1st AI and Security Symposium (co-sponsored by Tsinghua University and Alibaba Security), invited by Alibaba, Beijing.
- *2019.10*, The 5th GeekPwn International Security Geek Contest, Shanghai.


Translated with DeepL.com (free version)
<span class='anchor' id='-internships'></span>

# 💻 实习
- *2023.10 - 2024.10*, Palo Alto Lab, PAII, Inc.
- *2023.04 - 2023.06*, Fuxi Lab, Netease.
<!-- - *2022.11 - 2023.09*, TouTu lab, Tencent.-->
- *2022.06 - 2022.12*, 2012 Lab, Huawei.




**Thank you very much for every visitor, and we look forward to hearing from you!**

<script type="text/javascript" id="mapmyvisitors" src="//mapmyvisitors.com/map.js?d=rKMwhJZp-jNdf9O9kF5nNmH24oOX225WsWhZMH3I8bQ&cl=ffffff&w=a"></script>