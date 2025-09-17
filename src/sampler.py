import os

class ParentSampler:
    """Sampler to gather families within a batch"""

    def __init__(self, outcomes_dict, batch_size, sampler):
        self.outcomes_dict = outcomes_dict
        self.batch_size = batch_size
        self._sampler = sampler

    def __iter__(self):
        current_batch = []
        n = 0

        for idx in self._sampler:
            pnrs = [self.outcomes_dict["person_id"][idx]]

            if (parents := self.outcomes_dict["parents"][idx]) is not None:
                parent_pnrs = [parent["parent_id"] for parent in parents]
                pnrs.extend(parent_pnrs)

            n += len(pnrs)
            if n > self.batch_size:
                yield current_batch
                n = len(pnrs)
                current_batch = []
            current_batch.append(pnrs)

        if current_batch:
            yield current_batch


class UnpadSampler:
    """Sampler to gather people based on seq_lens"""

    def __init__(self, lengths, n_tokens, max_seq_len, sampler):
        self.lengths = lengths
        self.n_tokens = n_tokens
        self.max_seq_len = max_seq_len
        self._sampler = sampler
        self._debug = os.environ.get("DEBUG_DATA") == "1"

    def __iter__(self):
        current_batch = []
        current_tokens = 0
        batch_id = 0

        for idx in self._sampler:
            length = int(min(self.lengths[idx], self.max_seq_len))

            if length > self.n_tokens:
                if current_batch:
                    if self._debug:
                        print(f"[Sampler] yield batch {batch_id} size={len(current_batch)} tokens={current_tokens}")
                    yield current_batch
                    batch_id += 1
                    current_batch = []
                    current_tokens = 0
                if self._debug:
                    print(f"[Sampler] oversize single idx={idx} len={length} (budget {self.n_tokens})")
                yield [idx]
                batch_id += 1
                continue

            if current_tokens + length > self.n_tokens:
                if current_batch:
                    if self._debug:
                        print(f"[Sampler] yield batch {batch_id} size={len(current_batch)} tokens={current_tokens}")
                    yield current_batch
                    batch_id += 1
                current_batch = [idx]
                current_tokens = length
            else:
                current_batch.append(idx)
                current_tokens += length

        if current_batch:
            if self._debug:
                print(f"[Sampler] final batch {batch_id} size={len(current_batch)} tokens={current_tokens}")
            yield current_batch