# Deep Learning Approaches for Image Captioning: A Hybrid CNN-Transformer Framework

**Authors:** Ali Hamza, Zarmeena Jawad  
**Registration Numbers:** B23F0063AI106, B23F0115AI125  
**Department:** Department of Artificial Intelligence  
**Institute:** Pak-Austria Fachhochschule Institute of Applied Sciences and Technology  
**Course:** Digital Image Processing (COMP-342)  
**Instructor:** Mr. Hussan Ali Shah  
**Submission Deadline:** 17 April 2026, 11:59 PM

---

## Working Note

This Markdown draft integrates the core content from **Assignment 1 (Literature Review)** and **Assignment 2 (Proposed Methodology)** into the structure required for the **final research paper**.  
Sections such as **Introduction**, **Literature Review**, and **Methodology** are already drafted.  
Sections that depend on actual implementation and experiments, especially **Implementation Details**, **Results and Analysis**, **Conclusion**, and the **final Abstract**, must be updated after running the model and collecting results.

---

## 1. Abstract

> Write this section after completing implementation and results analysis.

### Draft Abstract Template

Image captioning is a challenging task in digital image processing that requires a model to understand visual content and generate meaningful natural language descriptions. This paper presents a hybrid **CNN-Transformer** framework for image captioning that combines **ResNet-50** for visual feature extraction with a lightweight **Transformer decoder** for caption generation. The model is trained and evaluated on the **MS COCO** dataset using standard image captioning metrics including **BLEU**, **METEOR**, **CIDEr**, and **SPICE**. The proposed approach is designed to improve semantic alignment between image content and generated captions while maintaining lower computational complexity than large-scale end-to-end Transformer architectures. Experimental results should be summarized here after implementation, including the final values for the main evaluation metrics and a short comparison with baseline methods. The study aims to show that an efficient hybrid architecture can provide a practical balance between caption quality, model size, and computational cost for real-world image captioning applications.

---

## 2. Introduction

Image captioning is the task of automatically generating textual descriptions that accurately represent the content of an image. It is a significant research problem in digital image processing because it combines visual understanding with natural language generation. This task lies at the intersection of **computer vision** and **natural language processing**, requiring a model to identify objects, understand scene relationships, and produce grammatically correct and semantically meaningful descriptions.

The problem is important because image captioning has many real-world applications. It can support **assistive technologies for visually impaired users**, improve **image retrieval systems**, enhance **human-computer interaction**, and contribute to **multimedia indexing** and **autonomous systems**. A strong image captioning system must not only recognize objects in an image but also capture context, interactions, and scene-level meaning.

Earlier image captioning methods relied on handcrafted visual features and rule-based or template-based sentence generation. These approaches had limited generalization ability and could not handle the diversity of real-world images. Later, deep learning methods improved performance by using **Convolutional Neural Networks (CNNs)** for feature extraction and **Recurrent Neural Networks (RNNs)** for caption generation. Although CNN-RNN frameworks improved caption quality, recurrent models still suffered from limited capacity to model long-range dependencies in language sequences.

Transformer-based architectures addressed many of these limitations by introducing **self-attention mechanisms**, which allow the model to capture long-range dependencies more effectively and align visual and textual features more precisely. However, modern large Transformer-based captioning models often require extensive pretraining data, large computational resources, and high memory usage. These constraints make them difficult to deploy in smaller research environments and resource-constrained systems.

This research is motivated by the need to design an image captioning system that balances **caption quality**, **semantic accuracy**, and **computational efficiency**. The proposed work therefore focuses on a **hybrid CNN-Transformer architecture**, where a pretrained **ResNet-50** extracts image features and a lightweight **Transformer decoder** generates captions. The main contribution of this work is the design of a computationally efficient framework that aims to maintain strong captioning performance while reducing model complexity compared with full Transformer-based encoder-decoder architectures.

---

## 3. Literature Review

Recent image captioning research has increasingly focused on **CNN-Transformer** and **vision-language pretraining** approaches because of their strong ability to jointly model visual and textual information. The following studies provide the main academic basis for this project.

### 3.1 BLIP

Li et al. proposed **BLIP (Bootstrapping Language-Image Pre-training)**, a framework that uses a bootstrapping strategy for vision-language pretraining with noisy web image-text pairs. The model combines a visual encoder with a Transformer-based text generation component and demonstrates strong captioning performance on the **MS COCO** dataset. The study showed improvements in caption fluency and semantic relevance, but it also highlighted the requirement for large-scale pretraining data and substantial computational resources.

### 3.2 GIT

Wang et al. introduced **GIT (Generative Image-to-Text Transformer)**, a unified Transformer-based framework for image captioning and related vision-language generation tasks. The approach achieved strong performance on benchmark datasets, particularly in terms of **CIDEr** and **SPICE** scores. However, the large model size and high training cost make it difficult to use in real-time and resource-limited settings.

### 3.3 BLIP-2

Li et al. later proposed **BLIP-2**, which improves efficiency by connecting frozen pretrained image encoders and large language models through a lightweight querying Transformer. BLIP-2 significantly reduces the number of trainable parameters while maintaining competitive captioning performance. Despite this advantage, its reliance on frozen pretrained components may reduce flexibility for domain-specific adaptation.

### 3.4 Attention-Enhanced Transformer Models

Chen et al. explored attention-enhanced Transformer-based captioning methods that better capture spatial relationships among objects in an image. Their work showed that improved attention mechanisms can produce more semantically accurate and visually grounded captions. However, the model still struggled with rare objects and unfamiliar visual concepts.

### 3.5 CNN-Vision Transformer Hybrid Models

Zhang et al. proposed hybrid architectures that use CNNs to capture local visual details and Transformers to model global context. Their results on **MS COCO** and **Flickr30k** showed improvements in caption diversity and sentence coherence. The main limitation of these approaches is increased architectural complexity and longer training time.

### 3.6 Lightweight Transformer Approaches

Kumar et al. focused on lightweight Transformer-based captioning models designed for lower computational cost. These methods achieved competitive results relative to larger models, but qualitative analysis showed that generated captions were often less descriptive for complex scenes with multiple objects and interactions.

### 3.7 Comparative Analysis

| Paper | Year | Method | Dataset | Metrics | Main Strength | Main Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Li et al. (BLIP) | 2022 | CNN + Transformer | MS COCO | BLEU, CIDEr | High caption quality | High compute cost |
| Wang et al. (GIT) | 2022 | Transformer-based | MS COCO | CIDEr, SPICE | Strong text generation | Large model size |
| Li et al. (BLIP-2) | 2023 | Querying Transformer | MS COCO | CIDEr | Efficient training | Depends on frozen pretrained modules |
| Chen et al. | 2023 | Attention Transformer | MS COCO | BLEU, METEOR, CIDEr | Better semantic alignment | Weak on rare objects |
| Zhang et al. | 2024 | CNN + ViT hybrid | MS COCO, Flickr30k | CIDEr, BLEU | Diverse captions | Longer training time |
| Kumar et al. | 2024 | Lightweight Transformer | MS COCO | BLEU, CIDEr | Lower computation | Less descriptive captions |

### 3.8 Research Gap

The literature review identifies several limitations in current image captioning approaches:

1. Many high-performing models require large-scale pretraining and expensive computational resources.
2. Several methods still suffer from **caption hallucination**, where the generated caption is fluent but factually incorrect.
3. Lightweight models improve efficiency but may lose descriptive richness in complex scenes.
4. Existing evaluation metrics such as BLEU and CIDEr do not fully capture factual correctness or grounding between words and image regions.

These limitations motivate the development of a **hybrid CNN-Transformer framework** that aims to improve efficiency while preserving semantic quality and visual-textual alignment.

---

## 4. Methodology / Proposed Solution

### 4.1 Problem Definition

The goal of this research is to develop an image captioning system that can generate meaningful and contextually relevant textual descriptions for images while keeping the model computationally efficient. The proposed solution focuses on combining the strengths of **CNN-based visual feature extraction** with the sequence modeling power of a **Transformer decoder**.

### 4.2 Research Objective

The primary objective is to design a **hybrid CNN-Transformer image captioning model** that:

1. Preserves caption quality and semantic relevance.
2. Reduces model complexity compared with full Transformer encoder-decoder systems.
3. Maintains stronger alignment between image content and generated text.

### 4.3 Proposed Architecture

The proposed system contains two major components:

#### 4.3.1 Visual Feature Extraction Module

A pretrained **ResNet-50** model is used as the CNN backbone. The final classification layer is removed, and the network is used only to extract high-level visual features from the input image. The extracted **2048-dimensional feature vector** is then projected through a linear layer to a **512-dimensional embedding**, which is used as input to the caption generator.

**Dimensional flow:**  
`224 x 224 x 3 -> ResNet-50 -> 2048-d feature vector -> Linear Projection -> 512-d visual embedding`

#### 4.3.2 Caption Generation Module

The caption generation component is a lightweight **Transformer decoder** that takes the projected image embedding and the tokenized caption sequence as input. The decoder then predicts the next word token at each time step in an autoregressive manner.

**Transformer decoder configuration:**

- Number of decoder layers: 4
- Number of attention heads: 8
- Hidden size: 512
- Feed-forward dimension: 2048
- Dropout: 0.1

The output of the decoder is a probability distribution over the vocabulary, from which the next word is generated.

### 4.4 Mathematical View

Let an input image be represented by \( I \). The CNN encoder extracts a visual representation:

\[
v = f_{cnn}(I)
\]

where \( v \in \mathbb{R}^{2048} \). A linear projection transforms it into:

\[
z = Wv + b
\]

where \( z \in \mathbb{R}^{512} \). The Transformer decoder then models the probability of the caption sequence \( Y = (y_1, y_2, ..., y_T) \) conditioned on the image embedding:

\[
P(Y|I) = \prod_{t=1}^{T} P(y_t | y_{<t}, z)
\]

The training objective is to minimize the cross-entropy loss between the predicted and target caption tokens:

\[
\mathcal{L} = - \sum_{t=1}^{T} \log P(y_t^{*} | y_{<t}, z)
\]

where \( y_t^{*} \) is the ground-truth token at step \( t \).

### 4.5 Measurable Targets

The original measurable targets proposed for this study are:

- **CIDEr >= 110**
- **BLEU-4 >= 35**
- **At least 25% reduction in trainable parameters** compared with full Transformer encoder-decoder architectures

These targets should be revisited after final implementation and experimentation.

---

## 5. Implementation Details

> Update this section with your actual implementation environment once experiments are complete.

### 5.1 Tools and Libraries

The proposed implementation is expected to use the following tools:

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **NLTK / Hugging Face tokenizers / transformers** for caption tokenization and sequence handling

If other libraries are used during implementation, add them here.

### 5.2 Hardware Used

Replace this subsection with the actual hardware used during training and evaluation.

**Example format:**

- CPU: `[Insert CPU model]`
- GPU: `[Insert GPU model, if used]`
- RAM: `[Insert RAM capacity]`
- Operating System: `[Insert OS]`

### 5.3 Dataset Description

The model is trained and evaluated on the **MS COCO** dataset, which is one of the most widely used benchmarks for image captioning.

**Dataset characteristics:**

- Approximately 120,000 images
- Five human-written captions per image
- Real-world scenes with multiple objects and interactions
- Standard annotations available in JSON format

**Why MS COCO was selected:**

1. It is the standard benchmark used by many image captioning studies.
2. It contains diverse and complex scenes.
3. It supports standardized comparison with prior work.

### 5.4 Data Preprocessing

The following preprocessing pipeline was proposed in Assignment 2 and should be reported here if used in the final implementation:

#### Image preprocessing

- Resize images to `224 x 224`
- Convert images to tensors
- Normalize using ImageNet mean and standard deviation

#### Data augmentation for training

- Random horizontal flip with probability `0.5`
- Random rotation within `+/-15 degrees`
- Color jitter for brightness and contrast with factor `0.2`
- Random resized crop with scale range `0.8-1.0`

#### Caption preprocessing

- Convert all captions to lowercase
- Remove punctuation
- Apply tokenization
- Add special tokens: `<SOS>`, `<EOS>`, `<PAD>`
- Pad or truncate captions to a fixed length of `30` tokens

### 5.5 Training, Validation, and Testing Split

Use the actual split used in your experiments. If you follow the common MS COCO split, mention it clearly here.

**Draft split format:**

- Training set: `[Insert number of images]`
- Validation set: `[Insert number of images]`
- Test set: `[Insert number of images]`

### 5.6 Hyperparameters

The proposed training configuration is:

| Component | Configuration |
| --- | --- |
| Loss function | Cross-entropy loss with `<PAD>` ignored |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Scheduler | Cosine annealing |
| Batch size | `32` |
| Epochs | `20-30` |
| Early stopping | Patience = `5` |

### 5.7 Constraints and Practical Challenges

This subsection should describe the actual implementation challenges you faced, such as:

- Limited GPU memory
- Long training time
- Data loading bottlenecks
- Caption tokenization issues
- Overfitting
- Caption hallucination
- Rare object recognition failure

If you made changes to the original architecture or hyperparameters, explain them here.

---

## 6. Results and Analysis

> Do not invent values here. Replace all placeholders after running experiments.

### 6.1 Quantitative Results

Insert the final numerical performance of your model using the evaluation metrics relevant to image captioning.

**Suggested results table:**

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr | SPICE |
| --- | --- | --- | --- | --- | --- |
| Baseline model | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Proposed CNN-Transformer model | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

If you run multiple experiments, include mean and standard deviation over repeated runs.

### 6.2 Comparison with Related Work

Compare the proposed method with at least one baseline or related method from the literature.

**Suggested comparison table:**

| Method | Dataset | BLEU-4 | CIDEr | Key Observation |
| --- | --- | --- | --- | --- |
| CNN-RNN baseline | MS COCO | `[ ]` | `[ ]` | `[ ]` |
| Lightweight Transformer baseline | MS COCO | `[ ]` | `[ ]` | `[ ]` |
| Proposed hybrid CNN-Transformer | MS COCO | `[ ]` | `[ ]` | `[ ]` |

### 6.3 Qualitative Results

Include sample predictions from the trained model.

**Suggested qualitative analysis format:**

| Image ID | Ground Truth Caption | Predicted Caption | Observation |
| --- | --- | --- | --- |
| `[Image 1]` | `[ ]` | `[ ]` | `[ ]` |
| `[Image 2]` | `[ ]` | `[ ]` | `[ ]` |
| `[Image 3]` | `[ ]` | `[ ]` | `[ ]` |

You may also include:

- Attention heatmaps
- Grad-CAM visualizations
- Failure case examples

### 6.4 Discussion

This subsection should analyze the results rather than only reporting them.

You should discuss:

1. Whether the proposed model met the target metrics.
2. Whether the model produced semantically meaningful captions.
3. Whether the model reduced computational complexity compared with larger architectures.
4. In which situations the model performed well.
5. In which situations the model failed.

### 6.5 Strengths

Possible strengths to confirm or revise after experiments:

- Lower complexity than large end-to-end Transformer systems
- Better balance between efficiency and caption quality
- Stronger semantic alignment than simpler baselines
- Easier deployment in constrained environments

### 6.6 Weaknesses and Failure Cases

Possible issues to analyze after experiments:

- Rare object omission
- Incorrect object relationships
- Caption hallucination
- Short or generic captions
- Reduced detail in complex scenes

---

## 7. Conclusion and Future Work

### 7.1 Draft Conclusion

This study proposes a structured hybrid **CNN-Transformer** framework for image captioning that combines **ResNet-50** for visual feature extraction with a lightweight **Transformer decoder** for caption generation. The framework is designed to balance caption quality and computational efficiency by reducing model complexity while maintaining semantic relevance in generated captions. Based on the literature review and proposed methodology, the approach is academically justified and practically suitable for evaluation on the **MS COCO** benchmark.

This conclusion must be revised after implementation to include the actual performance achieved, whether the measurable targets were met, and what aspects of the system worked better or worse than expected.

### 7.2 Future Work

Possible future directions include:

1. Integrating stronger visual encoders such as EfficientNet or Vision Transformers.
2. Using parameter-efficient fine-tuning methods for larger pretrained captioning systems.
3. Applying grounding-aware learning to reduce caption hallucination.
4. Evaluating the model on domain-specific datasets beyond MS COCO.
5. Using reinforcement learning or contrastive objectives to improve semantic accuracy.

---

## 8. References

> Convert these into strict IEEE style in Word before final submission.

1. I. A. Albadarneh, B. H. Hammo, and O. S. Al-Kadi, "Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation," *Computer Science Review*, vol. 58, p. 100766, 2025.
2. S. Assad, N. A. M. Isa, and S. A. M. Saleh, "Hybrid CNN-Transformer Models for Industrial Defect Detection: A Systematic Review," *Results in Engineering*, 2026.
3. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *arXiv preprint arXiv:2010.11929*, 2021.
4. J. Li, D. Li, C. Xiong, and S. Hoi, "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in *Proceedings of the 39th International Conference on Machine Learning*, 2022, pp. 12888-12900.
5. J. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models," *arXiv preprint*, 2023.
6. R. Mokady, A. Hertz, and A. H. Bermano, "ClipCap: CLIP Prefix for Image Captioning," *arXiv preprint arXiv:2111.09734*, 2021.
7. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proceedings of the 38th International Conference on Machine Learning*, 2021, pp. 8748-8763.
8. D. Sharma, R. Dingliwal, C. Dhiman, and D. Kumar, "Lightweight Transformer with GRU Integrated Decoder for Image Captioning," 2022.
9. O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, "Show and Tell: A Neural Image Caption Generator," *arXiv preprint arXiv:1411.4555*, 2015.
10. J. Wang et al., "GIT: A Generative Image-to-Text Transformer for Vision and Language," 2022.

---

## 9. Final Submission Checklist

- Revise the abstract after final results are available.
- Replace all placeholder tables in the Results and Analysis section.
- Add actual hardware and software details used during implementation.
- Insert architecture diagrams, sample outputs, and captions for all figures.
- Convert references to full IEEE format in the final Word file.
- Keep plagiarism below 20%.
- Export the final version into Word using the required template.
- Add a public GitHub or Google Drive link to the implementation code.

