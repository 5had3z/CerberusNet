#!/usr/bin/env python

__author__ = "Bryce Ferenczi"
__email__ = "bryce.ferenczi@monashmotorsport.com"

from trainer_base_class import ModelTrainer

class MonoSegmentationTrainer(ModelTrainer):
    def __init__(self, model, optimizer, loss_fn, dataloaders, learning_rate=1e-4, savefile=None, checkpoints=True):
        '''
        Initialize the Model trainer giving it a nn.Model, nn.Optimizer and dataloaders as
        a dictionary with Training, Validation and Testing loaders
        '''
        super().__init__(model, optimizer, loss_fn, dataloaders, learning_rate, savefile, checkpoints)
    

    def _calculate_accuracy(self, fx, y):
        preds = torch.argmax(fx,dim=1,keepdim=True)
        correct = preds.eq(y.view_as(preds)).sum().item()
        acc = correct/y.nelement()
        return acc

    def _intersectionAndUnion(self, imPred, imLab):
        """
        This function takes the prediction and label of a single image,
        returns intersection and union areas for each class
        To compute over many images do:
        for i in range(Nimages):
            (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
        """
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred = torch.argmax(imPred,dim=1,keepdim=True).squeeze(dim=1)
        imPred = imPred.cpu().detach().float()
        imLab = imLab.cpu().float()

        imPred = imPred * (imLab >= 0).float()
        numClass = 3

        # Compute area intersection:
        intersection = imPred * (imPred == imLab).float()
        (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection
        acc = area_intersection / area_union
        print(area_intersection)
        print(area_union)
        print(acc)
        return acc

    def visualize_output(self):
        """
        Forward pass over a testing batch and displays the output
        """
        test_batch = iter(self._testing_loader)

        with torch.no_grad():
            self._model.eval()
            image, mask = next(test_batch)
            image = image.to(self._device)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)/test_batch.index_sampler.batch_size

            pred = torch.argmax(output[0],dim=1,keepdim=True)
            for i in range(test_batch.index_sampler.batch_size):
                plt.subplot(1,3,1)
                plt.imshow(np.moveaxis(image[i,:,:,:].cpu().numpy(),0,2))
                plt.xlabel("Base Image")
        
                plt.subplot(1,3,2)
                plt.imshow(mask[i,:,:])
                plt.xlabel("Ground Truth")
        
                plt.subplot(1,3,3)
                plt.imshow(pred.cpu().numpy()[i,0,:,:])
                plt.xlabel("Prediction")

                plt.suptitle("Propagation time: " + str(propagation_time))
                plt.show()

    def custom_image(self, filename):
        """
        Forward Pass on a single image
        """
        with torch.no_grad():
            self._model.eval()

            image = Image.open(filename)

            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize([1024,2048]),
                transforms.ToTensor()
            ])
            image = img_transform(image).unsqueeze(0)
            image = image.to(self._device)

            print(image.shape)

            start_time = time.time()
            output = self._model(image)
            propagation_time = (time.time() - start_time)

            pred = torch.argmax(output[0],dim=1,keepdim=True)

            plt.subplot(1,2,1)
            plt.imshow(np.moveaxis(image[0,:,:,:].cpu().numpy(),0,2))
            plt.xlabel("Base Image")
        
            plt.subplot(1,2,2)
            plt.imshow(pred.cpu().numpy()[0,0,:,:])
            plt.xlabel("Prediction")

            plt.suptitle("Propagation time: " + str(propagation_time))
            plt.show()