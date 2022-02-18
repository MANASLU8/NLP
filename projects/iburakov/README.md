Данный проект является результатом выполнения практических заданий курса.

**Студент:** Бураков Илья Алексеевич<br>
**Платформа:** Python 3.9

## Структура проекта

* `assets` - вспомогательные и конфигурационные файлы, не являющиеся исходным кодом.
    * `raw-dataset` - **незакоммиченная директория**, куда следует поместить
      распакованный [исходный датасет](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) для запуска проекта.
    * `requirements.txt` - файл с зависимостями.
* `source` - исходный код.
    * `task1` - результат выполнения лабораторной работы №1.
    * `tests` - модульные тесты.

## Настройка окружения

Необходим интерпретатор Python 3.9 с установленными пакетами из `assets/requirements.txt`. Рекомендуется использовать
`virtualenv`/`venv`.

## Запуск проекта

```shell
cd projects/iburakov
export PYTHONPATH=source  # or $env:PYTHONPATH="source" in PowerShell
python -m task1 
```

## Запуск тестов

```shell
cd projects/iburakov
export PYTHONPATH=source  # or $env:PYTHONPATH="source" in PowerShell
python -m pytest source/tests 
```