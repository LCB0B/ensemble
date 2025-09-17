# %%
class Animal:
    def speak(self) -> str:
        return "Some generic sound"


class Mammal(Animal):
    def announce(self) -> str:
        """
        Calls speak, intended to be overridden.
        """
        return f"The mammal says: {self.speak()}"


class Dog(Mammal):
    def speak(self) -> str:
        return "Woof!"


# %%

# Example usage
d = Dog()
print(d.announce())  # Output: The mammal says: Woof!

# %%
import polars as pl

from src.paths import FPATH

outcomes_path = (
    FPATH.NETWORK_DATA / "socmob_full_sample" / f"gpa_targets_transformer"
).with_suffix(".parquet")
# %%
df = pl.read_parquet(outcomes_path)
# %%
outcomes_path
# %%
