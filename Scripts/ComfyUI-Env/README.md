# ComfyUI Runtime Environment Manager

## 🇺🇦 Опис (Ukrainian)

Цей PowerShell скрипт автоматизує розгортання та оновлення середовища ComfyUI.  
Він дозволяє легко керувати Python, venv, Git та Microsoft Visual C++ Runtime.

### Важливо

- Скрипт **запускається через `ComfyUI-Env.bat`**, який повинен лежати в тій же папці, де знаходиться скрипт.  
- Всі дії відбуваються **в тій же папці**, тобто ComfyUI, venv та інші файли створюються там же.

### Як він працює:

1. **Перший запуск**  
   - Встановлює Python Launcher, Git, VC++ Runtime, обрану гілку Python та створює venv.  
   - Клонує репозиторій ComfyUI.  

2. **Повторний запуск (оновлення)**  
   - Перевіряє оновлення Git та Python Launcher.  
   - Виконує мінорні оновлення встановленої версії Python.  
   - Підтримує Pip в актуальному стані.  
   - Папки ComfyUI та venv залишаються без змін.  

3. **Зміна гілки Python**  
   - Для зміни гілки видаліть папку `venv`.  
   - Скрипт запропонує обрати нову гілку Python, створить чистий venv.  
   - Папка ComfyUI залишається недоторканою, всі локальні зміни збережені.  

### Особливості:

- Не перевстановлює ComfyUI, якщо папка вже існує.  
- Підтримка трьох останніх гілок Python для вибору.  
- Summary report показує реально встановлені версії Python, Git, VC++ та Pip.  
- Стару версію Python видаляє автоматично, якщо існує файл `python_version.txt`.  

---

## 🛠 ComfyUI Runtime Environment Manager – Повний Flow (Українська)

```mermaid
flowchart TD
    A[Запуск скрипта через ComfyUI-Env.bat] --> B{Чи існує папка venv?}

    %% Перший запуск
    B -- Ні та python_version.txt відсутній --> C[Вибір гілки Python]
    C --> D[Встановлення обраної версії Python]
    D --> E[Створення нового venv]
    E --> F[Клонування ComfyUI, якщо немає]
    F --> G[Оновлення Pip]
    G --> H[Запис обраної гілки в python_version.txt]
    H --> I[Summary: реальні версії Python, Git, VC++, Pip]

    %% Повторний запуск (оновлення)
    B -- Так & гілка не змінювалась --> J[Перевірка мінорних оновлень Python]
    J --> K[Оновлення Pip]
    K --> L[Перевірка оновлень Git]
    L --> I

    %% Зміна гілки Python
    B -- Так & venv видалено --> M[Вибір нової гілки Python]
    M --> N[Встановлення нової гілки Python]
    N --> O[Створення нового venv]
    O --> P[Оновлення Pip]
    P --> I
	
---

## 🇬🇧 Description (English)

This PowerShell script automates the deployment and updating of the ComfyUI runtime environment.  
It simplifies managing Python, venv, Git, and Microsoft Visual C++ Runtime.

### Important

- The script **is run via `ComfyUI-Env.bat`**, which must be in the same folder as the script.  
- All actions occur **in that folder**, so ComfyUI, venv, and other files are created there.

### How it works:

1. **First run**  
   - Installs Python Launcher, Git, VC++ Runtime, the selected Python branch, and creates a venv.  
   - Clones the ComfyUI repository.  

2. **Subsequent run (updates)**  
   - Checks for Git and Python Launcher updates.  
   - Applies minor updates to the installed Python version.  
   - Keeps Pip up to date.  
   - ComfyUI and venv folders remain untouched.  

3. **Changing Python branch**  
   - To switch Python branch, delete the `venv` folder.  
   - The script will prompt to select a new Python branch and create a fresh venv.  
   - ComfyUI folder is untouched; all local changes are preserved.  

### Features:

- Does not reinstall ComfyUI if the folder exists.  
- Supports choosing from the latest three Python branches.  
- Summary report displays the actual installed versions of Python, Git, VC++, and Pip.  
- Old Python version is automatically removed if `python_version.txt` exists.  

---

## 🛠 ComfyUI Runtime Environment Manager – Full Flow

```mermaid
flowchart TD
    A[Start Script via ComfyUI-Env.bat] --> B{Does venv folder exist?}

    %% First run
    B -- No & python_version.txt not exists --> C[Prompt: Select Python branch]
    C --> D[Install selected Python]
    D --> E[Create new venv]
    E --> F[Clone ComfyUI if not exists]
    F --> G[Upgrade Pip]
    G --> H[Write selected branch to python_version.txt]
    H --> I[Summary: real Python, Git, VC++, Pip versions]

    %% Subsequent run (updates)
    B -- Yes & no branch change --> J[Check for minor Python updates]
    J --> K[Upgrade Pip]
    K --> L[Check for Git updates]
    L --> I

    %% Change Python branch
    B -- Yes & venv deleted --> M[Prompt: Select new Python branch]
    M --> N[Install new Python branch]
    N --> O[Create fresh venv]
    O --> P[Upgrade Pip]
    P --> I