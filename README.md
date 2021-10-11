# Robot Infant Intention Recognition

Practical from my Social Signal Processing class.

The goals were:
  * To build features vectors based on speech signal's energy and f0
  * To train binary/3-classes SVM to detect speaker's intention on two different datasets (Kismet and BabyEars)
  * To train classifiers using a multi-corpus approach
  * To detect the type of interlocutor (robot or infant)

Regarding intention detection, classification accuracies are significaly higher on the Kismet dataset.
The binary classifier trained on *attention* and *prohibition* classes even reaches a detection rate of 100 % on a test set of size 142.

Multi-corpus approaches leads to poorer results. One can hypothesize that because humans are not used to speak to robots, they might exegerate their intentions thus leading to difference in terms of expressiveness between in the two datasets.  

<img src="/Image/kismet-photo3-full.jpg" width="512" height="384">

Kismet robot (https://robots.ieee.org/robots/kismet/)
