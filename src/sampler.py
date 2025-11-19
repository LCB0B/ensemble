class UnpadSampler:
    """Sampler to gather people based on seq_lens"""

    def __init__(self, lengths, n_tokens, max_seq_len, sampler):
        self.lengths = lengths
        self.n_tokens = n_tokens
        self.max_seq_len = max_seq_len
        self._sampler = sampler

    def __iter__(self):
        current_batch = []
        n = 0

        for idx in self._sampler:
            length = self.lengths[idx]
            length = int(min(length, self.max_seq_len))

            n += length
            if n > self.n_tokens:
                yield current_batch
                n = length
                current_batch = []
            current_batch.append(idx)

        if current_batch:
            yield current_batch
