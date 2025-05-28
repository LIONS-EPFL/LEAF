import os
import json
from sacrebleu.metrics import BLEU

import open_clip_pez


def compute_token_accuracy(reconstructions_ids, references_ids):
    n_correct = 0
    n_total = 0
    for rec, ref in zip(reconstructions_ids, references_ids):
        rec = [r for r in rec if r != 0]
        ref = [r for r in ref if r != 0][1:-1]  # remove bot/eot tokens
        assert len(rec) == len(ref)
        n_correct += sum([t in rec for t in ref])
        n_total += len(rec)
    return n_correct / n_total

def compute_word_accuracy(reconstructions, references):
    n_correct = 0
    n_total = 0
    for rec, ref in zip(reconstructions, references):
        rec = rec.lower().split()
        ref = ref.lower().split()
        n_correct += sum([t in rec for t in ref])
        n_total += len(rec)
    return n_correct / n_total


def compute_bleu_score(reconstructions, references):
    pass



if __name__ == '__main__':
    ref_captions_path = "coco_captions.txt"
    reconstuctions_dir = "/mnt/cschlarmann37/project_bimodal-robust-clip/results_inversions"

    with open(ref_captions_path, 'r') as f:
        refs = f.readlines()
    refs = [r.strip() for r in refs]
    bleu = BLEU(references=[[r] for r in refs])

    for reconstructions_file in os.listdir(reconstuctions_dir):
        if not reconstructions_file.startswith("results-") or not reconstructions_file.endswith(".json"):
            continue
        print(f"\nevaluating {reconstructions_file}")
        reconstructions_path = os.path.join(reconstuctions_dir, reconstructions_file)
        res_dict = json.load(open(reconstructions_path, 'r'))
        args = res_dict["config"]
        res_list = res_dict["results"]
        refs_ = [r["original"] for r in res_list]
        reconstructions = [r["reconstructed"] for r in res_list]
        orig_ids = [r["ids_orig"] for r in res_list]
        reconstructions_ids = [r["ids_rec"] for r in res_list]
        assert refs_ == refs[:len(refs_)]

        sim = [r["sim"] for r in res_list]
        sim_avg = sum(sim) / len(sim)
        print(f"Average cos-sim: {sim_avg:.4f}")

        word_acc = compute_word_accuracy(reconstructions, refs)
        print(f"Word accuracy: {word_acc * 100:.4f}")

        token_acc = compute_token_accuracy(reconstructions_ids, orig_ids)
        print(f"Token accuracy: {token_acc*100:.4f}")

        score = bleu.corpus_score(reconstructions, references=None)
        print(f"BLEU score: {score.score:.4f}")

        # print latex row
        print(f"{sim_avg:.2f} & {word_acc*100:.1f} & {token_acc*100:.1f} & {score.score:.1f}")
