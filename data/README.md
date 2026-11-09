# Data

The main dataset is **MealRec+** (Li et al., SIGIR 2024), downloaded from [WUT-IDEA/MealRecPlus](https://github.com/WUT-IDEA/MealRecPlus) at pinned commit `8bd7d7d7effb505be4a7d2074b28abad802fe0f7` (2024-07-28). The upstream repository states no licence, so **the data are not redistributed here**. `data/raw/` is git-ignored.

```bash
bash scripts/download_mealrecplus.sh            # relation files + healthiness scores (~5 MB)
bash scripts/download_mealrecplus.sh --meta     # also course.zip, user_course.zip (Git LFS, ~400 MB)
bash scripts/download_mealrecplus.sh --verify   # compare SHA-256 against data/SHA256SUMS
```

## Layout (`data/raw/MealRec+{H,L}/`)

| File | Contents |
|---|---|
| `user_course.txt` | `user \t course`. **Not split**: it also contains the courses of meals in the test set (the leak). |
| `meal_course.txt` | `meal \t course`. Every meal has exactly 3 courses. |
| `course_category.txt` | `course \t category` (0 appetizer, 1 main, 2 dessert). One line per **meal–course edge** (H has 11,451 lines but only 942 courses), not one line per course as the upstream README says. |
| `user_meal_{train,tune,test}.txt` | `user \t meal`, random 8:1:1 split. |
| `healthiness/{course,meal,user}_{fsa,who}.txt` | One score per line. FSA in 4–12 (lower is healthier), WHO in 0–7 (higher is healthier). A user's score is the mean over that user's **training** meals (original split; checked in `tests/test_data.py`). |
| `meta_data/{user,course}2index.txt` | `raw id \t index`. |
| `meta_data/course.zip` → `course.csv` | Name, category, rating, image, ingredients, directions, nutrition (per serving, as a Python dict), reviews, tags. |
| `meta_data/user_course.zip` → `user_course.csv` | `user_id, course_id, rating (4–5 only), dateLastModified` (2000-02 to 2018-03). |
| `reference/` | Upstream `data_load.py` and `healthiness_eval.py`, kept for comparison. |

## Statistics

| | MealRec+H | MealRec+L |
|---|---:|---:|
| Users / meals / courses | 1,575 / 3,817 / 7,280 | 1,928 / 3,578 / 10,589 |
| Courses that appear in at least one meal | 942 | 1,956 |
| User–course edges | 151,148 | 181,087 |
| User–meal train / tune / test | 37,413 / 4,677 / 4,677 | 9,445 / 1,181 / 1,181 |

[`SHA256SUMS`](SHA256SUMS) lists the SHA-256 of every file. The hash of `course.zip` matches the upstream Git LFS oid.

## Additional datasets

| Directory | Source | Download |
|---|---|---|
| `bundle_crosscbr/datasets/{Youshu,NetEase,iFashion}/` | `dataset.tgz` in [mysbupt/CrossCBR](https://github.com/mysbupt/CrossCBR), commit `bbe80dfb` (SHA-256 `95dcb76d…b1be2`) | `curl -L -o data/raw/bundle_crosscbr/dataset.tgz https://github.com/mysbupt/CrossCBR/raw/bbe80dfb1bbe878d72389185edefc655c8b5db45/dataset.tgz`, then `tar xzf` in the same directory |
| `foodcom/RAW_{interactions,recipes}.csv` | Food.com (Majumder et al., EMNLP 2019), Kaggle `shuyangli94/food-com-recipes-and-user-interactions` | `kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions -f RAW_interactions.csv -p data/raw/foodcom` (same for `RAW_recipes.csv`), then `unzip` |

Loader: `mealrec.data.load_bundle` reads the bundle datasets (meal = bundle, course = item). Food.com has no loader; see `experiments/g1_foodcom_profile.py`.
