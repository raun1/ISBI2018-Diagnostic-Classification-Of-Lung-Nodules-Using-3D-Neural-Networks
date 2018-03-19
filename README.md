# LIDC_2018_LUNG_CLASSIFICATION
Network Architecture for the ISBI_2018 paper : DIAGNOSTIC CLASSIFICATION OF LUNG NODULES USING 3D NEURAL NETWORKS 
## Built With/Things Needed to implement

* [Python](https://www.python.org/downloads/) - Python-2 
* [Keras](http://www.keras.io) - Deep Learning Framework used
* [Numpy](http://www.numpy.org/) - Numpy
* [Sklearn](http://scikit-learn.org/stable/install.html) - Scipy/Sklearn/Scikit-learn
* [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) - CUDA-8
* [CUDNN](https://developer.nvidia.com/rdp/assets/cudnn_library-pdf-5prod) - CUDNN-5 You have to register to get access to CUDNN


![alt text](https://github.com/raun1/LIDC_2018_LUNG_CLASSIFICATION/blob/master/images/architecture.PNG)
```
The code in this repository provides only the stand alone code for this architecture. You may implement it as is, or convert it into modular structure
if you so wish. The dataset of LIDC-IDRI can obtained from the link above and the preprocessiong steps involved are mentioned in the paper. (Link to be provided soon)
You have to provide the inputs.
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
The following lines provide a very simple example of how you may train the above network and obtain predictions.
You may include validation set and callback function as demonstrated by the lines marked with ******** however please note we never used those
In the following lines the network is named as "finalmodel"
Apply early stopping if desired, for our experiments we didnt.

xyz=keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto') <---------- early stopping, set patience to the number of epochs you want the network to continue before stopping,in case the performance starts to decline



The inputs are X1_train,X2_train
finalmodel.fit([X1_train,X2_train], [y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train],  
                batch_size=8, #I used a batch size of 8, compared to batch size of 2-16, 8 gave the best result, higher batch size couldnt be supported on our GPU's
                nb_epoch=150,
                validation_data=([X2_validate,X3_validate],[y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate]), 
                shuffle=True,
                 callbacks=[xyz], *********
                class_weight=class_weightt)
You can obtain predicitions of the different outputs using the keras's model.predict function
Set the index variable from 0-8 to obtain corresponding outputs with 8 being the final output
predict_x=finalmodel.predict([X2_test,X3_test],batch_size=8)[index]
```

## Contribution

All the code were written by the authors of this paper.
Please contact (raun- rd31879@uga.edu) for questions and queries

## Things to cite -
If you use Keras please cite it as follows - 
###### @misc{chollet2015keras,title={Keras},author={Chollet, Fran\c{c}ois and others},year={2015},publisher={GitHub},howpublished={\url{https://github.com/keras-team/keras }},}
### If you borrow the concept of multi_output only then cite our paper - To be declared.... (after April 2018)

