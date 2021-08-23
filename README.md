# Listen-Attend-Spell
<b>Implementation</b>

Implemented in TensorFlow 2.3.0 directly from the paper (<a href=https://arxiv.org/abs/1508.01211>Chan, et al. 2015</a>).

My version and the paper version underwent substantially different training.  The version in the paper was trained on an internal dataset containing 3 million utterances, increased a further 20 times using a room simulator, for 2 weeks on 32 DistBelief replicas.  My version was trained on 28 thousand utterances from the <a href=https://www.openslr.org/12>LibriSpeech</a> corpus, with no room simulator, for 8 hours on a gaming laptop.

The differences in training may explain the difference in results.  While the paper obtained a 14.1% word error rate, decoding on a separate test set, my system achieves a 52.8% character accuracy rate, on validation data when (mostly) teacher-forcing the previous character.

Example of next character prediction when teacher forcing the previous character:

<pre>
predicted : 'and hy te wauld nover hslow tn  of tis hause  ah te art oh teph a staang '
target    : 'anyway he would never allow one of his horses to be put to such a strain$'
</pre>

Example of pure decoding, from logmels and the start-of-sentence token:

<pre>
decoded : 'and the state of the state of the state of the state of the state of the state of the state$'
target  : 'anyway he would never allow one of his horses to be put to such a strain$'
</pre>

The decoder example suggests that the language model of the decoder is currently dominating the acoustics of the listener.

My code is available in <a href=https://github.com/redonovan/Listen-Attend-Spell/blob/main/listenattendspell.py>listenattendspell.py</a>, with TensorBoard training curves in <a href=https://github.com/redonovan/Listen-Attend-Spell/blob/main/TensorBoard.png>TensorBoard.png</a>; 1 epoch = 3546 steps.
