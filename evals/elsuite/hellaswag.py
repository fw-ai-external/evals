from .multiple_choice import MultipleChoice, get_dataset, Sample
import evals
import evals.metrics
from evals.formatting import make_abc
from evals.record import RecorderBase


class HellaSwag(MultipleChoice):
    def eval_sample(self, sample, rng, few_shots):
        assert isinstance(sample, Sample)

        options, correct_answer = make_abc(
            answers=sample.answers,
            correct_idx=sample.label,
            rng=rng,
        )

        prompt = (
            few_shots + sample.activity_label + ": " + sample.question + "\n" + options
        )

        result = self.completion_fn(
            prompt=prompt,
            multiple_choices=["A", "B", "C", "D"],
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        sampled = result.get_completions()[0]

        evals.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=correct_answer,
        )

    def run(self, recorder: RecorderBase):
        samples = get_dataset(self.dataset)
        few_shots = []
        for i in range(10):
            few_shots.append(
                samples[i].activity_label + ": " + samples[i].shot + "\n\n"
            )
        few_shots = "".join(few_shots)
        self.eval_all_samples(recorder, samples, few_shots)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
