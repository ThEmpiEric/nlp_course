# Train 

#Librerias ...
 

# Funciones utilis
# Obtener una distibucion de probabilidade de los logits 
def get_preds(raw_logits): 
    '''
    Funcion para obetener la distribucion de probabilidad de los logits del modelo 
    '''
    probs = F.softmax(raw_logits.detach(), dim = 1)
    y_pred = torch.argmax(probs, dim = 1).cpu().numpy()
    return y_pred


def model_eval(data,model, gpu = False):
    with torch.no_grad(): 
        preds, trgs = [], [] 
        for window_words, labels in data: 
            if gpu:
                window_words = window_words.cuda()

            outputs = model(window_words)

            # Get predictions 
            y_pre = get_preds(outputs)
            preds.append(y_pre)

            trg  = labels.numpy()
            trgs.append(trg)
    # Aplanar el batch 
    tgts = [e for l in trgs for e in l]
    preds = [e for l in preds for e in l]
            
    return accuracy_score(tgts,preds)


def save_checkpoint(state, is_best, checkpoint_patn, filename = 'checkpoint.pt'):
    '''Funcion para serializar objetos a disco y guardar ese objeto'''
    filename = os.path.join(checkpoint_patn, filename)
    torch.save(state, filename)
    if is_best: 
        shutil.copyfile(filename, os.path.join(checkpoint_patn, 'model_best.pt'))


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                num_epochs, patience, use_gpu, savedir, get_preds, model_eval, save_checkpoint):

    start_time = time.time()
    best_metric = 0
    metric_history = []
    train_metric_history = []
    n_no_improve = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        loss_epoch = []
        training_metric = []
        model.train()

        for window_words, labels in train_loader:
            # Mover datos a GPU si está disponible
            if use_gpu:
                window_words = window_words.cuda()
                labels = labels.cuda()

            # Forward pass
            outputs = model(window_words)
            loss = criterion(outputs, labels)
            loss_epoch.append(loss.item())

            # Obtener métricas de entrenamiento
            y_pred = get_preds(outputs)
            tgt = labels.cpu().numpy()
            training_metric.append(accuracy_score(tgt, y_pred))

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Métrica promedio en el conjunto de entrenamiento para la época actual
        mean_epoch_metric = np.mean(training_metric)
        train_metric_history.append(mean_epoch_metric)

        # Evaluación en el conjunto de validación
        model.eval()
        tuning_metric = model_eval(valid_loader, model, gpu=use_gpu)
        metric_history.append(tuning_metric)

        # Actualizar scheduler basado en la métrica de validación
        scheduler.step(tuning_metric)

        # Verificar mejora en la métrica de validación
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        # Guardar checkpoint si hay mejora
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric
        }
        save_checkpoint(checkpoint, is_improvement, savedir)

        # Early stopping
        if n_no_improve >= patience:
            print("No improvement. Breaking out of loop.")
            break

        print('Train acc: {:.4f}'.format(mean_epoch_metric))
        print('Epoch [{}/{}], Loss: {:.4f} - Val accuracy: {:.4f} - Epoch time: {:.2f}s'
              .format(epoch+1, num_epochs, np.mean(loss_epoch), tuning_metric, (time.time() - epoch_start_time)))

    print("--- Total training time: {:.2f} seconds ---".format(time.time() - start_time))
    return best_metric, metric_history, train_metric_history