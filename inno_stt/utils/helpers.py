import torch
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from hypersp.metrics.wer import word_error_rate
from typing import List, Tuple, Dict


def print_once(msg):
    if (not torch.distributed.is_initialized() or
        (torch.distributed.is_initialized() and
         torch.distributed.get_rank() == 0)):
        print(msg)


def add_ctc_labels(labels):
    if not isinstance(labels, List):
        raise ValueError("labels must be a list of symbols")
    labels.append("<BLANK>")
    return labels


def __ctc_decoder_predictions_tensor(tensor, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels) - 1  # id of a blank symbol
        for idx, p in enumerate(prediction):
            if p != previous and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


def __beamsearch_decoder_predictions_tensor(prob_tensor, pred_tensor, labels):
        """
        Takes output of greedy ctc decoder and performs ctc decoding algorithm to
        remove duplicates and special symbol. Returns prediction
        Args:
            tensor: model output tensor
            label: A list of labels
        Returns:
            prediction
        """
        
        blank_id = len(labels) - 1
        hypotheses = []
        labels_map = dict([(i, labels[i]) for i in range(len(labels))])
        batched_decoded = []
         
        # iterate over batch
        for ind in range(prob_tensor.shape[0]):
            prediction = pred_tensor[ind].numpy().tolist()
            prob = prob_tensor[ind].numpy().tolist()
      
            # CTC decoding procedure
            previous = blank_id  # id of a blank symbol
            decoded = []
            for idx, p in enumerate(prediction):
                if p[0] != previous and p[0] != blank_id:
                    if len(p) > 1 and p[1] == blank_id and prob[idx][1] > -1:
                        continue

                    decoded.append([ind, idx, [labels_map[c]
                                               for c in p], prob[idx]])

                previous = p[0]

            batched_decoded.append(decoded)
        #print('batched_decoded : ', batched_decoded)
        for ind in range(len(batched_decoded)):
            tmp = []
            for val in batched_decoded[ind]:
                tmp.append(val[2][0])
            hypotheses.append(f"{''.join(tmp).strip()}")                   
#        print('hypotheses : ', hypotheses)
        return hypotheses


def monitor_asr_train_progress(tensors: List, labels: List):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints wer and prediction examples to screen
    Args:
        tensors: A list of 3 tensors (predictions, targets, target_lengths)
        labels: A list of labels

    Returns:
        word error rate
    """
    references = []

    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    with torch.no_grad():
        targets_cpu_tensor = tensors[1].long().cpu()
        tgt_lenths_cpu_tensor = tensors[2].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            references.append(reference)
        hypotheses = __ctc_decoder_predictions_tensor(
            tensors[0], labels=labels)
    tag = "training_batch_WER"
    wer, _, _ = word_error_rate(hypotheses, references)
    print_once('{0}: {1}%'.format(tag, round(wer, 2) * 100))
    print_once('Prediction: {0}'.format(hypotheses[0]))
    print_once('Reference: {0}'.format(references[0]))
    return wer


def __gather_losses(losses_list: List) -> List:
    return [torch.mean(torch.stack(losses_list))]


def __gather_predictions(predictions_list: List, labels: List) -> List:
    results = []
    for prediction in predictions_list:
        results += __ctc_decoder_predictions_tensor(prediction, labels=labels)
    return results

def __gather_predictions_beamsearch(predictions_list: List, labels: List) -> List:
    results = []
    dict_obj = predictions_list[0]
    results += __beamsearch_decoder_predictions_tensor(dict_obj['prob_cpu_tensor'],dict_obj['pred_cpu_tensor'], labels=labels)
    return results

def __gather_predictions_beamsearch_lm(predictions_list: List) -> List:
    results = []
    predictions_list = predictions_list[0]
    for prediction in  predictions_list:
        results.append( prediction[0][1])
    return results



def __gather_transcripts(transcript_list: List,
                         transcript_len_list: List,
                         labels: List) -> List:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            results.append(reference)
    return results


def process_evaluation_batch_greedy(tensors: Dict,
                             global_vars: Dict,
                             labels: List):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(
                v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)

def process_evaluation_batch_beamsearch(tensors: Dict,
                             global_vars: Dict,
                             labels: List):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions_beamsearch(
                v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)
def process_evaluation_batch_beamsearch_lm(tensors: Dict,
                             global_vars: Dict,
                             labels: List):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):            
            global_vars['predictions'] += __gather_predictions_beamsearch_lm(v)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)



def process_evaluation_epoch(global_vars: Dict, tag=None, use_cer=False):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        global_vars: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    if 'EvalLoss' in global_vars:
        eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    else:
        eloss = None
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']
    
    wer, scores, num_words = word_error_rate(
        hypotheses=hypotheses,
        references=references,
        use_cer=use_cer)
    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= torch.distributed.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()
            del eloss_tensor

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        del scores_tensor
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        del num_words_tensor
        wer = scores * 1.0 / num_words
    return wer, eloss


def norm(x):
    if not isinstance(x, List):
        if not isinstance(x, Tuple):
            return x
    return x[0]


def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)


def model_multi_gpu(model, multi_gpu=False):
    if multi_gpu:
        model = DDP(model)
        print('DDP(model)')
    return model


def __load_module_func(module_name):
    mod = __import__(f'{module_name}', fromlist=[module_name])
    return mod


def dynamic_import(import_path: str):
    if ":" not in import_path:
        raise ValueError(
            "import_path should include path and class name"
            "ex) tasks.SpeechRecognition.librispeech.data.manifest:Manifest")
    module_name, object_name = import_path.split(":")
    module = __load_module_func(module_name)
    return getattr(module, object_name)
