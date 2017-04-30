#HSLIDE
# Text Summarization using Deep Learning

#HSLIDE
## Prima Focus
* Major Focus for this project is on extractive text summarization.
* This involves extracting sentences as it is from the given text to form a summary.
* General approach followed(before Deep Learning became a craze) was to find the saliency of sentences using hand engineered features.
* In this project we aim to shift to a more data centric model where features as well are learned from data.

#HSLIDE
Approaches studied till now(cited later) use:
* Convolutional Neural Network(CNN)
* Recurrent Nerual Network
	* More specifically LSTMs(Long Short Term Memory) [LSTMs by Colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

We have decided to move forward with a CNN based approach for now.

#HSLIDE
As initial part of model building process Word Vectors and consequently Paragraph vectors were studied.
* Word Vectors convert each word into n-dimensional vector by training them on prediction task.
* These vectors capture certain properties specific to the task they were trained for.
* They also exhibit certain simple vector operations to language very well.
Post studying about them a simple model using the proposed was built.

#VSLIDE
* Currently I am working on building a CNN-based model for Sentence catagorization into:
	* Sentence should belong to summary
	* Sentence should not belong to summary

```python
	class SentenceCNN(object):
    	#Define Init Function
    		def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,
                	 filter_sizes,num_filters,hidden_states,l2_reg=0.0):
```
PS: Explain structure from [diagram](https://github.com/deepvyas/Presentations/blob/MidSemPresentations/model1.png) and [diagram](http://aclweb.org/anthology/D/D14/D14-1181.pdf) for further illustration.

* For providing ground truths for training we plan to use the para-phrase system for labelling the sentences as extractive summaries as gold-labels are not available.
* For doing so the above mentioned model was used for training the para-phrase detector and it was used to generate the labels for each sentence.

#HSLIDE
* In the implemented model the the above shown three convolutional layers compute the convolution using filters of sizes 2,3 and 4 respectively. 
* This means that the filters compute features considering 2,3 and 4 words of sentences at a time.
* The pooling layers perform a max-over-time pooling which basically select the maximum feature value from each vector as its output.
* Dropout layer is to ensure that the model does not memorize the dataset which it often can given such large number of parameters. 

#VSLIDE
* This leads a poorly generalised model and Dropout is one of many approaches used to address this issue.
######Issues:
* One issue that we face with this is the problem of over-fitting which might be because the model is much more complex than the
amount of corpus available. Currently, to deal with this situation the complexity of the model was reduced.
* In the mean time search for a bigger corpus especially in Indian languages,as well as annotation of prior exsisting is being carried out.

#HSLIDE
* A basic LSTM layer with depth 32 stacked above the CNN described above was tried out.
* At current checkpoint it does not present much of a improvement due to various factors like:
	* Model is too complex for the little data we have.
	* It is not tuned much as training RNN is a tedious job and also this basic model does no good to us.
	* A better model involves using techniques like autoencoders with feedback mechanisms like attention mechanism.
#HSLIDE
#Flaw discussion
* The first and foremost flaw in following data driven approach is lack of sufficient amount of data.
* One other reason for the lack in performance of the model is the dependency of the model on the paraphrasing part.
* That part in itself is not the state-of-the-art.
#VSLIDE
* One other reason is the fact that only sentence level scoping is not that good an approximate.
* Also the depth of the network needs to be incresed if only data driven approach is the future goal.
* The above point again collapses to the point of lack of sufficient data.
#HSLIDE
###Future possible work:
* Collect a larger corpus so as to tune the current basic to its fullest and at the same time think upon newer models.
* One possible issue with above model could be that it is at a very rudimentary level able to capture document level features.
* For improving that we can convolve not just over sentences but over documents.
* Also could be used is an RNN-based approach wherein more focus is on the part on building models like Attention based models.
#HSLIDE
## References

* [Convolutional Neural Networks for Sentence Classification](http://aclweb.org/anthology/D/D14/D14-1181.pdf) by Yoon Kim
* [Extraction of Salient Sentences from Labelled Documents](https://arxiv.org/pdf/1412.6815.pdf) by Misha Denil,Alban Demiraj,Nando de Freitas
* [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/pdf/1603.07252.pdf) by Jianpeng Cheng Mirella Lapata 
* [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) by Quoc Le,Tomas Mikolov
