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

During my undergraduate years, I frequently participated in algorithm competitions. I was a member of the Alibaba Security Student Expert Group. I am ranked the Top 10 in the [Alibaba Security Challenger Program](https://s.alibaba.com/challenge?spm=a2c22.12281976.0.0.46db2a69WaN1Te).

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



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">IJCAI 2024 (CCF-A)</div><img src='images/CEAM.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Dialogue Cross-Enhanced Central Engagement Attention Model for Real-Time Engagement Estimation]() \\
Jun Yu, **Keda Lu**, Ji Zhao et al. (primary student author)

1. Propose center-based sliding window to solve the problem of repetitive inference in sliding windows, improving inference efficiency by 100%.
2. Propose the central engagement attention model based on SA, surpassing previous SOTA BiLSTM model, with inference efficiency improved by 300%.
3. Propose cross-enhanced module based on CA and seamlessly integrated with the core engagement attention model, establish a new SOTA result.

<!-- [**Project**](https://portaspeech.github.io/) \| [![](https://img.shields.io/github/stars/NATSpeech/NATSpeech?style=social&label=Code+Stars)](https://github.com/NATSpeech/NATSpeech) \| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/NATSpeech/PortaSpeech) -->
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024 (CCF-A) workshop</div><img src='images/mvav.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[MvAV-pix2pixHD: Multi-view Aerial View Image Translation]() \\
Jun Yu, **Keda Lu**, Shenshen Du et al. (primary student author)

1. Time-priority sampling and random sampling methods are designed for multi-view image translation tasks.
2. MvAV-pix2pixHD is proposed for multi-view image translation, using three powerful losses.
3. This method won the 1st and 2nd place in the MAVIC-T competition for two multi-view image translation tasks.

<!-- [**Project**](https://portaspeech.github.io/) \| [![](https://img.shields.io/github/stars/NATSpeech/NATSpeech?style=social&label=Code+Stars)](https://github.com/NATSpeech/NATSpeech) \| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/NATSpeech/PortaSpeech) -->
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACM-MM 2023 (CCF-A)</div><img src='images/sw2.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

- `ACM-MM 2023(CCF-A)` [Sliding Window Seq2seq Modeling for Engagement Estimation](https://dl.acm.org/doi/abs/10.1145/3581783.3612852) \\
Jun Yu, **Keda Lu**, Mohan Jing et al. (primary student author)

- `TOMM 2024在投(CCF-B)` [Exploring Seq2seq Models for Engagement Estimation in
Dyadic Conversations]() \\
Jun Yu, **Keda Lu**, Lei Wang et al. (primary student author)

1. Design multiple Seq2seq engagement estimation model based on Transformer and BiLSTM architectures.
2. Propose sliding window to address the significant context loss issue.
3. Propose Ai-BiLSTM to align and interact multimodal features of dialogue participants, further enhancing performance.
4. This method won the championship🏆 at ACM-MM 2023.


<!-- [**Project**](https://portaspeech.github.io/) \| [![](https://img.shields.io/github/stars/NATSpeech/NATSpeech?style=social&label=Code+Stars)](https://github.com/NATSpeech/NATSpeech) \| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/NATSpeech/PortaSpeech) -->
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Trans 在投</div><img src='images/500x300.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[A Comprehensive and Unified Out-of-Distribution Classification Solution Framework]() \\
Jun Yu, **Keda Lu**, Yifan Wang et al. (primary student author)

1. Propose semantic masking data augmentation for enhancing model robustness against occlusion scenarios.
2. Propose OOD-DAS, a comprehensive and robust data augmentation collection.
3. Propose OOD-Attention, which seamlessly integrates with SOTA classification models to improve model robustness.
4. Propose an iterative pseudo-labeling method for ensemble integration of multiple architecture models, further enhancing OOD recognition accuracy.
5. This method won the championship🏆 at ICCV 2023.

</div>
</div>





- `ACM-MM 2023` [Answer-Based Entity Extraction and Alignment for Visual Text Question Answering](https://dl.acm.org/doi/abs/10.1145/3581783.3612850) Jun Yu, Mohan Jing, Weihao Liu, Tongxu Luo, Bingyuan Zhang, **Keda Lu** et al.


- `CLEF 2022` [Bag of Tricks and a Strong Baseline for FGVC.](https://ceur-ws.org/Vol-3180/paper-182.pdf) Jun Yu, Hao Chang, **Keda Lu** et al.


- `CLEF 2022` [Efficient Model Integration for Snake Classification](https://ceur-ws.org/Vol-3180/paper-181.pdf) Jun Yu, Hao Chang, Zhongpeng Cai, Guochen Xie, Liwen Zhang, **Keda Lu** et al.


- `CVPR 2022 workshop` [Pseudo-label generation and various data augmentation for semi-supervised hyperspectral object detection](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html) Jun Yu, Liwen Zhang, Shenshen Du, Hao Chang, **Keda Lu** et al.


- `AAAI 2022 workshop` [Mining limited data for more robust and generalized ML models](https://alisec-competition.oss-cn-shanghai.aliyuncs.com/competition_papers/20211201/rank10.pdf), Jun Yu, Hao Chang, **Keda Lu** et al.


- `International Journal of Machine Learning and Cybernetics` [Generating transferable adversarial examples based on perceptually-aligned perturbation](https://link.springer.com/article/10.1007/s13042-020-01240-1), Hongqiao Chen, **Keda Lu**, Xianmin Wang et al.



<span class='anchor' id='-projects'></span>

# 💻 Projects


- *2024.03 - now* Multimodal Large Language Models

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">EDA Show</div><img src='images/eda.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2023.10 - 2024.02* Loan Customer Repayment Intention Recognition

Conducted exploratory data analysis (EDA) on a dataset with millions of records and tens of millions of call transcripts.
EDA -> Data Cleaning -> Feature Engineering. Utilized BERT for text modeling to identify customers' repayment intentions.
Explored leveraging LLM for data augmentation on call transcripts to enhance model robustness.
</div>
</div>

- *2023.05 - 2023.09* Vertical Domain Chat Assistant (Training Corpus Construction, Based on ChatGLM, Bloomz, Qwen, etc., fine-tuning compared to Lora)



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">OCR Large Model Showcase Platform</div><img src='images/ocrdemo.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2023.03 - 2023.06* OCR Large Model Showcase Platform

1. Use Gradio for both front-end and back-end to construct the entire OCR large model showcase interface, incorporating DocQA, MLLM, and pure OCR modules.
2. Independently maintained for internal analysis and debugging, as well as external business showcasing. This project was awarded the 2023 H1 XXX·Enterprise Excellence Award - Technical Advancement.
3. Responsible for the DocQA module.


</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Chinese font generation</div><img src='images/font.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2023.01 - 2023.03* Chinese font generation of Arbitrary style (GAN、Diffusion model)


1. Research on Chinese font generation algorithms, including DG-Font and Diff-Font.
2. Collect a dataset of 400 font classes and design an end-to-end font generation model based on the Diffusion model (DDPM). It slightly outperformed Diff-Font and DG-Font in metrics such as SSIM and LPIPS.

- Future Improvements: End-to-end, Contrastive learning, Diffusion model.


</div>
</div>




<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Document generation and style transfer</div><img src='images/style.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2022.11 - 2023.01* Document generation and style transfer (Independent research)

1. Explore the possibility of Diffusion model and GAN for end-to-end document generation.
2. Research five years of style transfer articles from top conferences, CNN -> Attention -> Transformer, including AdaIN (ICCV2017), MetaNet (CVPR2018), SANet (CVPR2019), MAST (ACM-MM 2020), StyleFormer (ICCV2021), AdaAttN (ICCV2021) and StyTr2 (CVPR2022).
3. Reproduce StyTr2 (CVPR2022) and AdaAttN (ICCV2021) and transfer them to the document generation task for data augmentation.

- Future improvements: Contrastive learning, GAN, Diffusion model

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Face Recognition and Text Detection</div><img src='images/mindspore.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2022.06 - 2022.12* Reproducing mainstream algorithms based on the Mindspore algorithmic framework

1. Participated in reproducing the [RetinaFace face detection algorithm](https://github.com/mindspore-lab/mindface)
2. Independent reproduction of the [FCENet text detection algorithm](https://github.com/mindspore-lab/mindocr)
</div>
</div>



<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Course Management System</div><img src='images/class.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- *2020.12 - 2021.01* Genetic Algorithm-based Intelligent Timetabling - Course Management System (Individually implemented)

1. Use sqlite3 databaseand and choose Bootstrap-Flask for visualization. Implement a unified login interface for students, teachers, and department heads with distinct client interfaces.
2. Propose an intelligent timetabling algorithm and proposed a novel optimization objective function (utilizing course variance). Employed **genetic algorithms** for optimization in timetabling.
3. This project comprises over 2000 lines of Python code and 1000 lines of HTML code. It has been openly shared on my personal [blog](https://blog.csdn.net/weixin_43999137/article/details/113178364) and [Github]((https://github.com/wujiekd/Integrated-course-design-of-software-direction)).
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Student performance management system</div><img src='images/score.gif' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

- *2019.04 - 2019.06* Student performance management system based on MFC (C++) (Individually implemented).
1. Includes all basic functions (Create, Read, Update, Delete), as well as operations like import, save, and sorting.
2. The design was primarily inspired by the large login button interface of QQ, aiming to create a clear and clean user experience.
3. This project comprises over 10,000 lines of C++ code and has been open-sourced on my personal [blog](https://blog.csdn.net/weixin_43999137/article/details/91184179) and [Github](https://github.com/wujiekd/MFC-student-performance-management-system).

</div>
</div>


<span class='anchor' id='-cyjs'></span>



<span class='anchor' id='-competitions'></span>

# 🏅 Competitions

### Master phase (Main force)
---

- *2024.03* [CVPR 2024: Multi-modal Aerial View Image Challenge - Translation](https://codalab.lisn.upsaclay.fr/competitions/17224) (Top3 prize 2500$, Solo, **Runner up**🥈) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/17224#results) [[Paper]]


- *2023.10* [ICCV 2023: Out Of Distribution Generalization: Object Classification track](https://codalab.lisn.upsaclay.fr/competitions/14068#results) (Solo, **Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/14068#results) [[Paper在投]]


- *2023.10* [ICCV 2023: Out Of Distribution Generalization: Pose Estimation track](https://codalab.lisn.upsaclay.fr/competitions/14074#learn_the_details) (Solo, **Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/14074#results) [[Report]](https://www.ood-cv.org/reports/pose/ImageNet1k-1st.pdf)


- *2023.07* [ACM-MM 2023: Grand challenge, Engagement Estimation](https://multimediate-challenge.org/) (Solo, **Champion**🏆) [[LeaderBoard]](https://multimediate-challenge.org/LBs/LB_engagement/) [[Paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3612852) [[New]](https://cloud.tencent.com/developer/news/1167803)


- *2022.10* [ECCV 2022: Out Of Distribution Generalization Track-1: Object Classification](https://www.ood-cv.org/challenge.html) (Top3 prize 3300$, **Runner up**🥈) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/6781#results) [[Code]](https://github.com/wujiekd/ECCV2022-OOD-CV-Challenge-Classification-Track-2nd-USTC-IAT-United) 


- *2022.10* [ECCV 2022: Out Of Distribution Generalization Track-2: Object Detection](https://www.ood-cv.org/challenge.html) (Top3 prize 3300$, **Runner up**🥈) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/6784#results) [[Code]](https://github.com/wujiekd/ECCV2022-OOD-CV-Challenge-detection-Track-2nd-Place-Program) 


- *2022.05* [CVPR 2022: FGVC9 workshop FungiCLEF2022 challenge](https://sites.google.com/view/fgvc9/competitions/fungiclef2022) (**Runner up**🥈) [[LeaderBoard]](https://www.kaggle.com/c/fungiclef2022/LB) [[Code]](https://github.com/wujiekd/Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification) [[Paper]](https://ceur-ws.org/Vol-3180/paper-182.pdf) 


- *2022.03* [CVPR 2022: Multi-modal Aerial View Object Classification - SAR+EO](https://codalab.lisn.upsaclay.fr/competitions/1392) (Top3 prize 6000$, **Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/1392#results) [[Report]](https://arxiv.org/abs/2205.01920) [[New]](https://zhuanlan.zhihu.com/p/493603389)


- *2022.03* [CVPR 2022: Multi-modal Aerial View Object Classification - SAR](https://codalab.lisn.upsaclay.fr/competitions/1388) (Top3 prize 6000$, **Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/1388#results) [[Report]](https://arxiv.org/abs/2205.01920) [[New]](https://zhuanlan.zhihu.com/p/493603389)


### Master phase（Assistance）
---
- *2023.12* [ICCV 2023: WECIA - Caption Generation Challenge](https://eval.ai/web/challenges/challenge-page/2104/overview) (**Champion**🏆) [[LeaderBoard]](https://eval.ai/web/challenges/challenge-page/2104/LB/5203)


- *2023.07* [ACM-MM 2023: Visual Text Question Answering](https://visual-text-qa.github.io/) (**3rd**🥉) [[LeaderBoard]](http://vtqa-challenge.fixtankwun.top:20010/) [[Paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3612850)


- *2023.03* [CVPR 2023: Multi-modal Aerial View Imagery Challenges - Translation](https://codalab.lisn.upsaclay.fr/competitions/9968) (Top3 prize 2250$, **Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/9968#results) [[Paper]](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_8) 


- *2022.06* [CVPR 2022: Robustness in Sequential Data challenge](https://codalab.lisn.upsaclay.fr/competitions/2618#learn_the_details) (**Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/2618#results) [[Report]](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/USTC_IAT_first_solution_rose_challenge_22.pdf) [[New]](https://nelslip.ustc.edu.cn/2022/0608/c26914a562921/page.htm)


- *2022.03* [CVPR 2022: Semi-Supervised Hyperspectral Object Detection Challenge](https://codalab.lisn.upsaclay.fr/competitions/1752) (**Champion**🏆) [[LeaderBoard]](https://codalab.lisn.upsaclay.fr/competitions/1752#results) [[Paper]](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html)


### Bachelor phase
---

- *2022.08* [Computer Competition of China](https://algo.weixin.qq.com/) (Top10 prize 560,000¥, Solo, **National Second Prize**, Top30/3000+)  [[LeaderBoard]](https://algo.weixin.qq.com/) [[Code]](https://github.com/wujiekd/WeChat-Big-Data-Challenge-2022-National-Second-Prize-Top30) 


- *2022.01* [AAAI 2022: Data-Centric Robust Learning on ML Models](https://advml-workshop.github.io/aaai2022/) (Top10 prize 1000,000¥, Solo, **Rank 10/3692**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/531939/rankingList?lang=en-us) [[Code]](https://github.com/wujiekd/RTA-Iterative-Search-AAAI2022) [[Paper]](https://alisec-competition.oss-cn-shanghai.aliyuncs.com/competition_papers/20211201/rank10.pdf) 


- *2021.11* [OPPO Security AI Challenge - Face Recognition Attacks](https://security.oppo.com/challenge/home.html) (Top10 prize 600,000¥, Solo, **Rank 12/2000+**) [[LeaderBoard]](https://security.oppo.com/challenge/rank.html) [[Code]](https://github.com/wujiekd/Hot-restart-black-box-face-adversarial-attack) 


- *2021.03* [CVPR 2021：White-box Adversarial Attacks on ML Defense Models](https://aisecure-workshop.github.io/amlcvpr2021/) (Top10 prize 100,000¥, **Rank 20/1681**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/531847/rankingList) [[Code]](https://github.com/wujiekd/CVPR2021_ODI_BIM_Attack?spm=a2c22.21852664.0.0.7830775fHm2G8V)  [[Blog]](https://tianchi.aliyun.com/forum/post/208313)

- *2020.10* [Adversarial Attacks on forged images](https://tianchi.aliyun.com/competition/entrance/531812) (Top10 prize 2 million ¥, **Rank 6/1666**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/531812/rankingList)


- *2020.08* [Tencent Advertising Algorithm Competition](https://algo.qq.com/) (Top10 prize 100,000$,  **Rank 11/10000+**) [[Code]](https://github.com/wujiekd/2020-Tencent-advertising-algorithm-contest-rank11) [[Blog]](https://blog.csdn.net/weixin_43999137/article/details/107657517?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171314962516800222817673%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171314962516800222817673&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-107657517-null-null.142^v100^pc_search_result_base3&utm_term=%E8%85%BE%E8%AE%AF%E5%B9%BF%E5%91%8A%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9Bwujiekd&spm=1018.2226.3001.4187)


- *2020.04* [Used Car Trading Price Forecast](https://tianchi.aliyun.com/competition/entrance/231784) (Solo, Winner,  **Rank 13/2815**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/231784/rankingList) [[Code]](https://github.com/wujiekd/Predicting-used-car-prices) [[Blog]](https://tianchi.aliyun.com/forum/post/104728) 

- *2020.03* [Text Adversarial Attack Competition](https://tianchi.aliyun.com/competition/entrance/231762) (Top10 prize 68,000¥, **Rank 4/1666**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/231762/rankingList) [[Code]](https://github.com/wujiekd/NLP_Chinese_adversarial_attack) [[Blog]](https://tianchi.aliyun.com/forum/post/95886) 



- *2019.12* [ImageNet Adversarial Attack Competition](https://tianchi.aliyun.com/competition/entrance/231761) (Top10 prize 68,000¥, **Rank /1522**) [[LeaderB]](https://tianchi.aliyun.com/competition/entrance/231761/rankingList) [[Blog]](https://tianchi.aliyun.com/forum/post/87389)


- *2019.10* [GeekPwn2019 CAAD CTF Finals](https://geekcon.top/hof/zh/index.html) (Finals prize 100,000¥, **Rank 5th place in Finals**) [[LeaderBoard]](https://tianchi.aliyun.com/competition/entrance/231784/rankingList) [[New]](https://www.gzhu.edu.cn/info/1070/3803.htm) 


<span class='anchor' id='-honors'></span>

# 🎖 Honors and Awards
- *2023.11* Huawei Scholarship (Top 30th in the university)
- *2023.10* National Scholarship (Top 1% of graduate students)
- *2022.10* National Scholarship (Top 1% of graduate students)
- *2021.10* National Scholarship (Top 1% of undergraduate students)
- *2020.10* National Scholarships (Top 1% of undergraduate)


<span class='anchor' id='-educations'></span>

# 🎓 Educations
- *2022.09 - 2025.07*, <a href="https://www.ustc.edu.cn/"><img class="svg" src="/images/ustclogo.png" width="23pt"></a> University of Science and Technology of China, Computer Technology, Recommended Exemptions, Master's Degree
- *2018.09 - 2022.06*, <a href="https://www.gzhu.edu.cn/"><img class="svg" src="/images/gzhulogo.png" width="20pt"></a> Guangzhou University, Computer Science and Technology (1/591), Bachelor's Degree


<span class='anchor' id='-meetings'></span>

# 🏛️ Academic conferences
- *2024.03*, Mindspore AI Framework Industry Conference (organized by Huawei), invited by Huawei, Beijing.
- *2023.11*, 31st ACM International Conference on Multimedia, Ottawa, Canada.
- *2020.12*, The 1st AI and Security Symposium (organized by Tsinghua University and Alibaba Security), invited by Alibaba, Beijing.
- *2019.10*, The 5th GeekPwn International Security Geek Competition, Shanghai.


<span class='anchor' id='-internships'></span>

# 💻 Internships
- *2023.10 - 2024.10*, Palo Alto Lab, PAII, Inc.
- *2023.04 - 2023.06*, Fuxi Lab, Netease.
<!-- - *2022.11 - 2023.09*, TouTu lab, Tencent.-->
- *2022.06 - 2022.12*, 2012 Lab, Huawei.




**Thank you very much for every visitor, and we look forward to hearing from you!**

<script type="text/javascript" id="mapmyvisitors" src="//mapmyvisitors.com/map.js?d=rKMwhJZp-jNdf9O9kF5nNmH24oOX225WsWhZMH3I8bQ&cl=ffffff&w=a"></script>