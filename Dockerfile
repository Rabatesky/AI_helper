# Slim, а не alpine: в alpine другая системная библиотека (musl), под неё нет
# готовых сборок многих пакетов, и они компилируются из исходников. Выигрыш в
# размере не окупает лишних минут сборки и странных ошибок.
FROM python:3.12-slim

# PYTHONUNBUFFERED — иначе Python буферизует вывод, и `docker logs` показывает
#   пустоту, пока буфер не заполнится.
# PYTHONDONTWRITEBYTECODE — .pyc-файлы в контейнере не нужны, живёт он один запуск.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Зависимости ставим ДО копирования кода. Docker кэширует слои по содержимому:
# пока requirements.txt не менялся, этот слой берётся из кэша, и правка кода
# пересобирает образ за секунды, а не качает aiogram заново.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Свой пользователь вместо root. Если в боте когда-нибудь найдётся уязвимость,
# у нападающего не будет прав администратора даже внутри контейнера.
# UID 1000 — первый обычный пользователь в Linux, совпадает с типичным
# пользователем на сервере, что упрощает права на подключённую папку data.
RUN useradd --create-home --uid 1000 appuser

COPY app ./app

# Папку данных создаём заранее и отдаём её appuser: иначе контейнер, запущенный
# не от root, не сможет создать в ней файл базы.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Запуск модулем, а не файлом — иначе не работают импорты вида `from app.config`.
CMD ["python", "-m", "app.bot"]
