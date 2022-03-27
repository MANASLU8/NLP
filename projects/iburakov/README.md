Данный проект является результатом выполнения практических заданий курса.

**Студент:** Бураков Илья Алексеевич<br>
**Платформа:** Python 3.9

## Структура проекта

* `assets` - вспомогательные и конфигурационные файлы, не являющиеся исходным кодом.
    * `raw-dataset` - **незакоммиченная директория**, куда следует поместить
      распакованный [исходный датасет](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) для запуска проекта.
    * `annotated-corpus` - незакоммиченная директория с токенизированным датасетом (ЛР №1).
    * `annotated-corpus/tokens.tsv` - словарь токенов (ЛР №2).
    * `corrupted-dataset` - незакоммиченная директория для ЛР №2, куда следует поместить повреждённый тестовый датасет.
    * `misc` - прочие ассеты
      * `qwerty_distances.json` - предрассчитанные расстояния между клавишами на клавиатуре QWERTY.
    * `requirements.txt` - файл с зависимостями.
* `source` - исходный код.
    * `taskX` - результат выполнения лабораторной работы №X.
    * `tests` - модульные тесты.

## Настройка окружения

Необходим интерпретатор Python 3.9 с установленными пакетами из `assets/requirements.txt`. Рекомендуется использовать
`virtualenv`/`venv`.

## Запуск проекта

```shell
cd projects/iburakov
export PYTHONPATH=source  # or $env:PYTHONPATH="source" in PowerShell
python -m task1 
python source/task2/scripts/generate_dictionary.py
python source/task2/scripts/evaluate_spell_correction.py
```

## Запуск тестов

```shell
cd projects/iburakov
export PYTHONPATH=source  # or $env:PYTHONPATH="source" in PowerShell
python -m pytest source/tests 
```