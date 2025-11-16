---
layout: cv
permalink: /cv/
title: cv
nav: true
nav_order: 4
---

# Experience

<h3><img src="https://d2u1z1lopyfwlx.cloudfront.net/thumbnails/fcac3c13-3a6c-5909-a6f4-162ee3f71b63/3a06f831-ba1d-5f54-b203-52c2c00f69c8.jpg" alt="mirage" width="25" height="25"/> Mirage (FKA Captions) </h3>
<p style="text-align:left;">
    <em>Research Engineer, Member of Technical Staff</em>
    <span style="float:right;">
        July, 2023 - Present
    </span>
</p>
<ul>
    <li>Contributed to development of Mirage, a 13B parameter text+audio+image-to-video generation model. Designed novel architectural components, implemented distributed training pipeline on 1000+ H100 GPUs achieving 40% MFU, and developed context-parallel training for sequence lengths up to 150k tokens.</li>
    <li>Optimized Mirage Video inference achieving 4x faster startup times and 2x overall speedup through fp8 quantization, custom attention kernels, torch compilation, and inference-time caching with minimal quality degradation.</li>
    <li>Optimized Mirage Audio model, a 10B parameter text+audio-to-audio model, achieving 30% training throughput improvement and 4x inference speedup through efficient kernels, torch compilation, and HSDP while maintaining quality.</li>
    <li>Designed and implemented novel audio-to-landmarks architecture for lip-sync generation through extensive architecture exploration and ablation studies.</li>
    <li>Designed and implemented automated evaluation framework for continuous checkpoint assessment using pub/sub architecture, integrating metrics computation, Weights&Biases logging, and automated video artifact uploads to cloud storage.</li>
    <li>Evaluated GPU providers and made key infrastructure decisions enabling efficient multi-node training at scale.</li>
</ul>

<h3><img src="https://pbs.twimg.com/media/FBp56YOXIAEYRSO.jpg" alt="yandex" width="25" height="25"/> Yandex School of Data Analysis </h3>
<p style="text-align:left;">
    <em>ML course tutor assistant</em>
    <span style="float:right;">
        February, 2022 - June, 2023
    </span>
</p>
<ul>
    <li>Served as ML course tutor at Yandex School of Data Analysis, giving lectures on NLP and computer vision, and designing coursework on distributed LLM training (GPT-2XL, OPT-6.7B).</li>
</ul>

<h3>
    <img src="https://freesoft.ru.net/storage/images/775/7748/774792/774792_normal.png" alt="gradient" width="25" height="25"/>
    <img src="https://play-lh.googleusercontent.com/RujFpUpJZjc-d5bScOi-n-zs9ak4vTs_Y_bB1rJDdjLxpZsSilM67r49R2fwfuNneMc=w240-h480-rw" alt="gradient" width="25" height="25"/>
    Gradient & Persona: AI Photo & Video mobile editors </h3>
<p style="text-align:left;">
    <em>Computer Vision Engineer</em>
    <span style="float:right;">
        August, 2022 - May, 2023
    </span>
</p>
<ul>
    <li>Designed and implemented novel image encoding method for personalized Stable Diffusion generation, replacing expensive DreamBooth fine-tuning with single-shot encoding from few images. Approach enabled identity-preserving generation without model fine-tuning; similar methods were later published and widely adopted in 2024.</li>
    <li>Developed real-time GANs running on mobile devices at HD quality, 60fps while maintaining sub-2MB model size through aggressive quantization and architecture optimization.</li>
    <li>Optimized Stable Diffusion inference achieving 30% speedup through architectural modifications, custom modules, and efficient sampling strategies.</li>
    <li>Trained production models for re-aging, body reshaping (images and video), and body segmentation, iterating rapidly on novel architectures and dataset curation strategies.</li>
    <li>Deployed models to iOS using CoreML and server infrastructure using TorchScript, optimizing for both on-device and cloud inference.</li>
</ul>

<h3><img src="https://stripe-images.s3.us-west-1.amazonaws.com/works-with/57716240664220abbfc76ae713a23d1dbc152308" alt="itechart" width="25" height="25"/> iTechArt </h3>
<p style="text-align:left;">
    <em>Machine Learning Engineer</em>
    <span style="float:right;">
        February, 2022 - August, 2022
    </span>
</p>
<ul>
    <li>Designed and implemented an image classification service using the gRPC endpoint client/server architecture and the FastAPI framework.</li>
    <li>Utilized Uvicorn and Prometheus in conjunction with Docker and Supervisord to create a robust and scalable solution.</li>
    <li>Developed and implemented custom model architectures using C++, resulting in up to a 45\% reduction in model latency.</li>
    <li>Generated synthetic datasets to supplement real data, leading to an increase in model accuracy of up to 10%.</li>
    <li>Successfully distilled the CNN model into a model that was 3 times smaller while maintaining nearly identical evaluation metrics.</li>
</ul>
<ul>
    <li>Created and curated custom datasets from unstructured client data using Pandas and SQL.</li>
    <li>Trained numerous time-series models for demand forecasting, reducing forecast MAPE by 20%.</li>
    <li>Constructed production pipelines with AirFlow to convert raw data into a feature vector, feed it into the model, and forecast the product demand.</li>
</ul>

<h3><img src="https://magistral-russia.ru/wp-content/uploads/2022/06/yandex_znak.png" alt="yandex" width="25" height="25"/> Yandex </h3>
<p style="text-align:left;">
    <em>Software Engineer Intern</em>
    <span style="float:right;">
        May, 2021 - November, 2021
    </span>
</p>
<ul>
    <li>Developed rule-based and NLP-based solutions for affiliations parsing.</li>
    <li>Developed data annotation service with Flask framework. Wrapped it into Docker and deployed to the server.</li>
    <li>Developed similarity metrics for searhing similar logs.</li>
</ul>

# Publications

- **Seeing Voices: Generating A-Roll Video from Audio with Mirage** - A. Sundararaman, A. Adishesha, A. Jaegle, D. Bigioi, H. Song, J. Kyl, J. Mao, K. Lan, M. Komeili, S. Athar, S. Babayan, S. Beliasau, W. Buchwalter. *arXiv:2506.08279*, 2025. [[link]](https://arxiv.org/abs/2506.08279)
- **How to Fine-Tune Very Large Model if It Doesn't Fit on Your GPU** - Technical article on distributed training techniques for large language models. 600+ reactions, 2022. [[link]](https://bestasoff.medium.com/how-to-fine-tune-very-large-model-if-it-doesnt-fit-on-your-gpu-3561e50859af)

# Education
<h3><img src="https://pbs.twimg.com/media/FBp56YOXIAEYRSO.jpg" alt="ysda" width="25" height="25"/> Yandex School of Data Analysis </h3>
<p style="text-align:left;">
    <em>Master's degree level Machine Learning developer academic program</em>
    <span style="float:right;">
        September, 2020 - June, 2022
    </span>
</p>
<ul>
    <li>Relevant coursework: Efficient Deep Learning Systems, Reinforcement Learning, Computer Vision, NLP, Recommendation Systems.</li>
</ul>

<h3><img src="https://images.seeklogo.com/logo-png/43/1/bsu-belarusian-state-university-logo-png_seeklogo-432128.png" alt="bsu" width="25" height="25"/> Belarusian State University </h3>
<p style="text-align:left;">
    <em>Bachelor of Computer Science</em>
    <span style="float:right;">
        September, 2018 - August, 2022
    </span>
</p>
<ul>
    <li>Awarded a full scholarship and stipend by the government per entrance exam results.</li>
</ul>