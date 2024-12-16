# Repositorio proyecto *Magalí Pereyra* - Director: *Lucas Baldezzari*

## Objetivos

- Generar registros de EEG en personas sanas, sin patologías. Cada persona voluntaria deberá ejecutar o imaginar, dependiendo la sesión, la ejecución de movimiento de prensión y apertura de su mano izquierda y derecha.
- Procesar la señal de EEG para generar mapas topográficos que demuestren la presencia de patrones ERD y ERS asociados a los movimientos ejecutados o imaginados para cada voluntario.

## Presentación de estímulos

Para la presentación de estímulos se hará uso de los scripts de Python programados para el [Hackathon BCI v2](https://github.com/lucasbaldezzari/bcihack2).

Por otro lado, el software desarrollado será de utilidad para generar los archivos de eventos que nos permitan sincronizar y llevar un registro de qué acción realizó o imaginó la persona.

### Instalando dependencias

- Se recominedo utilizar [Miniconda](https://docs.anaconda.com/miniconda/) para instalar un ambiente con las dependencias de Python.
- Las dependencias pueden ser encontradas en el archivo *bcihack2_enviroment.yml*.
- Se recomienda instalar la versión liviana del [PQt5 Designer](https://build-system.fman.io/qt-designer-download)

#### Instalando el ambiente

Seguir los siguientes pasos.

1. Clonar o descargar el [repositorio](https://github.com/lucasbaldezzari/maguiproject) del proyecto.
2. Utilizando miniconda ejecutar desde el *cmd* y en la carpeta principal del repositorio clonado el siguiente comando *conda env create -f bcihack2_enviroment.yml*.
3. Testear que el ambiente fue creado correctamente ejecutando desde el *cmd* lo siguiente: *conda activate magui*.

Luego sería posible utilizar Visual Studio para ejecutar los scripts utilizando el ambiente creado.

---

Enjoy!