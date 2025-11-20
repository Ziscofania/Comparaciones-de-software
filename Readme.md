# CONFIGURACIONES DEL PROYECTO 

Instalar un entorno virtual 
```bash
source venv/bin/activate
```
```bash
. venv/bin/activate
```
Instalar los requeriments 
```bash 
pip install requirements.txt
```

esto abrira un navegador para poder visualizar el proyecto 

# Construcción de la imagen Docker

Desde la raíz del proyecto (donde está el Dockerfile), ejecutar:
``` bash
docker build -t optimizador:latest .
```
## Ejecutar aplicacion 
```bash
docker run -p 8501:8501 optimizador:latest
```
esto expondra los ports y se podra ingresar a la web de la app 

Usando volumenes usamos el sigueinte comando 
```bash
docker run --rm -p 8501:8501
     -v "$(pwd)":/app:Z     
     optimizador:latest
```