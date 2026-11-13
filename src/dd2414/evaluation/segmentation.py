"""Original Simple-BEV binary segmentation counts (sigmoid then round)."""

def official_counts(logits, target, valid):
    prediction = logits.sigmoid().round().bool()
    truth, mask = target.gt(0.5), valid.gt(0.5)
    return dict(tp=int((prediction & truth & mask).sum()),
                fp=int((prediction & ~truth & mask).sum()),
                fn=int((~prediction & truth & mask).sum()),
                tn=int((~prediction & ~truth & mask).sum()))
