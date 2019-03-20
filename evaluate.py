# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:24:18 2018

@author: quent
"""

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from train import get_split, load_batch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim


#================ DATASET INFORMATION ======================
log_dir = '/home/admin-pc/inception2-resnet/logs_7000'

log_eval = '/home/admin-pc/inception2-resnet/logs_7000_test_280'

dataset_dir = '/home/admin-pc/inception2-resnet/logs_test_280'#在原来代码的基础上，将test和train的tfrecords分成连个文件夹放，test和train数据互不影响

batch_size = 8

num_epochs = 4

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)

#=================== Model Evaluate =========================
def run():
    #Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('test', dataset_dir)
        #images, raw_images, labels = load_batch(dataset, batch_size = batch_size, is_training = False)
        images,labels = load_batch(dataset, batch_size = batch_size, is_training = False)

        #Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        #Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        recall, recall_update = tf.contrib.metrics.streaming_recall(predictions, labels)#添加1
        #spectivity,spectivity_update = tf.contrib.metrics.streaming_specificity_at_sensitivity(predictions,labels,recall_update,num_thresholds=200)
        precise, precise_update = tf.contrib.metrics.streaming_recall(predictions, labels)#添加2
        f1_score, f1_score_update = tf.contrib.metrics.f1_score(labels,predictions)#添加3
        auc, auc_update = tf.contrib.metrics.streaming_auc(predictions,labels,num_thresholds=200,curve='ROC')#添加4


        #metrics_op = tf.group([accuracy_update,recall_update,spectivity_update,precise_update,f1_score_update,auc_update])#改动1
        metrics_op = tf.group([accuracy_update,recall_update,precise_update,f1_score_update,auc_update])#改动1


        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step


        #Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value ,recall_value ,precise_value , f1_score_value ,auc_value = sess.run([metrics_op,
                                           global_step_op, accuracy,recall,precise,f1_score,auc])#改动2
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f , Recall(Sensitivity): %.4f ,Precise: %.4f, F1_Score: %.4f, Auc: %.4f (%.2f sec/step)',
                         global_step_count, accuracy_value, recall_value,precise_value, f1_score_value ,auc_value,time_elapsed)#改动3

            #return accuracy_value,recall_value,spectivity_value,precise_value,f1_score_value ,auc_value

            return accuracy_value, recall_value,precise_value, f1_score_value, auc_value


        #Define some scalar quantities to monitor
        tf.summary.scalar('Test_Accuracy', accuracy)
        tf.summary.scalar('Test_Recall(Sensitivity)', recall)
       # tf.summary.scalar('Test_Spectivity', spectivity)
        tf.summary.scalar('Test_Precise', precise)
        tf.summary.scalar('Test_F1_Score', f1_score)
        tf.summary.scalar('Test_Auc', auc)

        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = log_eval, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in range(int(num_steps_per_epoch * num_epochs)):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    logging.info('Current Streaming Recall(sensitivity): %.4f', sess.run(recall))
                    #logging.info('Current Streaming Spectivity: %.4f', sess.run(spectivity))
                    logging.info('Current Streaming Precise: %.4f', sess.run(precise))
                    logging.info('Current Streaming F1_score: %.4f', sess.run(f1_score))
                    logging.info('Current Streaming Auc: %.4f', sess.run(auc))

                #Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)


                #Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op = metrics_op, global_step = sv.global_step)

            #At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
            logging.info('Final Streaming Recall: %.4f', sess.run(recall))
            #logging.info('Final Streaming Spectivity: %.4f', sess.run(spectivity))
            logging.info('Final Streaming Precise: %.4f', sess.run(precise))
            logging.info('Final Streaming F1_score: %.4f', sess.run(f1_score))
            logging.info('Final Streaming Auc: %.4f', sess.run(auc))

            #Now we want to visualize the last batch's images just to see what our model has predicted
            raw_images, labels, predictions,probabilities = sess.run([images, labels, predictions,probabilities])
            TP = 0
            FN = 0
            FP = 0
            TN = 0

            for i in range(10):
                image, label, prediction,probability = raw_images[i], labels[i], predictions[i],probabilities[i][0]
                prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
                if (label_name == 'Normal'):#如果标签为0
                    if (prediction_name == 'Normal'):#如果预测为0
                        TP = TP + 1
                    else:
                        FN = FN + 1
                else :
                    if (prediction_name == 'Normal'):
                        FP = FP + 1
                    else:
                        TN = TN + 1
                if (label_name == 'Normal'):
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('0' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt', 'a')
                    f_score.write(str(probability) + '\n')
                    f_score.close()
                if (label_name == 'Abnormal'):
                    f_true = open('./y_true.txt', 'a')
                    f_true.write('1' + '\n')
                    f_true.close()

                    f_score = open('./y_scores.txt', 'a')
                    f_score.write(str(probability) + '\n')
                    f_score.close()

                text = 'Prediction: %s \n Ground Truth: %s' %(prediction_name, label_name)
                img_plot = plt.imshow(image)

                #Set up the plot and hide axes
                plt.title(text)
                img_plot.axes.get_yaxis().set_ticks([])
                img_plot.axes.get_xaxis().set_ticks([])
                plt.show()

            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
            print(TP, FN, FP, TN)

            y_test = TP + FP + FN + TN
            # correct_predictions = float(sum(all_predictions == y_test))
            # wrong_predictions = float(sum(all_predictions == y_test))
            precision = float(TP / (TP + FP))
            recall = float(TP / (TP + FN))
            Acc = float((TP + TN) / (TP + FP + FN + TN))
            TPR = float(TP / (TP + FN))  # sensitivity召回率 （TPR，真阳性率，灵敏度，召回率）
            TNR = float(TN / (FP + TN))  # specificity（TNR，真阴性率，特异度）

            FNR = float(FN / (TP + FN))  # 漏诊率，（1-sensitivity）
            FPR = float(FP / (TN + FP))  # 假正例率(1-specificity),假阳性率，误诊率
            # print(sum(all_predictions))
            print('\nTotal number of test examples: {}'.format(y_test))

            print('Accuracy: %.2f%%' % (Acc * 100))

            print('precision: %.2f%%' % (precision * 100))
            print('sensitivity/recall(TPR): %.2f%%' % (recall * 100))
            # print('sensitivity(TPR):%.2f%%' % (TPR*100))
            print('specificity(TNR): %.2f%%' % (TNR * 100))


if __name__ == '__main__':
    run()
