import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics


def tr_plot(tr_data, start_epoch):
    # Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    # plt.style.use('fivethirtyeight')
    plt.show()


def plot_metrics(model, history, test_gen):
    tr_plot(history, 0)
    acc = model.evaluate(test_gen, batch_size=32, steps=None, verbose=1)[1] * 100
    msg = 'Model accuracy on test set: ' + str(acc)
    print(msg, (0, 255, 0), (55, 65, 80))

    # Plotting train_loss vs val_loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()

    # Plotting train_accuracy vs Val_accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend(loc='lower right')


from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(model, X_test, y_test):
    print('test label shape', y_test.shape)
    print('test image shape', X_test.shape)
    print('Evaluate on test-data:')
    model.evaluate(X_test, y_test)

    pred = model.predict(X_test)

    bin_predict = np.argmax(pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Confusion matrix:
    matrix = confusion_matrix(y_test, bin_predict)
    print('Confusion Matrix:\n', matrix)
    return matrix


import itertools


# Plot the Confusion matrix:
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    sns.set(style="dark")
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n Accuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

def evaluate_model(model, test_gen, bin_predict):
    for X_batch, y_batch in test_gen:
        y_test = y_batch
        X_test = X_batch
        break

        matrix = compute_confusion_matrix(model, )

        plot_confusion_matrix(cm=np.array(matrix),
                              normalize=False,
                              target_names=['EarlyPreB', 'PreB', 'ProB', 'benign'],
                              title="Confusion Matrix")

        plot_confusion_matrix(cm=np.array(matrix),
                              normalize=True,
                              target_names=['EarlyPreB', 'PreB', 'ProB', 'benign'],
                              title="Confusion Matrix, Normalized")

        class_metrics = metrics.classification_report(y_test, bin_predict, labels=[0, 1])
        print(class_metrics)

        FP = matrix.sum(axis=0) - np.diag(matrix)
        FN = matrix.sum(axis=1) - np.diag(matrix)
        TP = np.diag(matrix)
        TN = matrix[:].sum() - (FP + FN + TP)

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        FDR = FP / (TP + FP)

        ACC = (TP + TN) / (TP + FP + FN + TN)

        print('Other Metrics:')
        MAE = mean_absolute_error(y_test, bin_predict)

        print('MAE ----------------------------------------------:', MAE)
        print('Accuracy -----------------------------------------:', ACC)
        print('Precision (positive predictive value)-------------:', PPV)
        print('Recall (Sensitivity, hit rate, true positive rate):', TPR)
        print('Specificity (true negative rate)------------------:', TNR)
        print('Negative Predictive Value-------------------------:', NPV)
        print('Fall out (false positive rate)--------------------:', FPR)
        print('False Negative Rate-------------------------------:', FNR)
        print('False discovery rate------------------------------:', FDR)

        preds = model.predict(X_test)
        # print(preds)
        print('Shape of preds: ', preds.shape)
        plt.figure(figsize=(12, 12))

        number = np.random.choice(preds.shape[0])

        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            number = np.random.choice(preds.shape[0])
            pred = np.argmax(preds[number])
            actual = (y_test[number])
            col = 'g'
            if pred != actual:
                col = 'r'
            plt.xlabel('N={} | P={} | GT={}'.format(number, pred, actual),
                       color=col)  # N= number P= prediction GT= actual (ground truth)
            image = X_test[number]  # cv2.cvtColor(X_test[number], cv2.COLOR_BGR2RGB)
            plt.imshow(((image * 255).astype(np.uint8)), cmap='binary')
        plt.show()
