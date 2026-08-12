# PhysioSentinel AI v15.5.5

## Cambio exclusivo: Control autonómico longitudinal progresivo

Se mantiene sin cambios el resto de pestañas y el motor XGBoost/Gradient Boosting, incluida la capa probabilística calibrada, el protocolo 70/15/15, Walk-Forward, persistencia y lógica de predicción.

### Corrección principal
La pestaña 10 ya no exige tres referencias para poder construir los vectores autonómicos. La referencia se construye sólo con registros ANTERIORES del mismo paciente y prioriza la misma fase.

- 0 registros previos: sin referencia longitudinal; sólo pueden usarse criterios absolutos disponibles y la confianza se reduce.
- 1 registro previo: cambio individual normalizado (Δ norm.) respecto al registro previo. No se presenta como Z estadístico.
- 2–4 registros previos: estandarización individual usando únicamente los registros anteriores.
- ≥5 registros previos: Z robusto individual mediante mediana/MAD, con SD como fallback.

### Protección temporal
El registro actual y los registros posteriores quedan excluidos de la referencia. De este modo, al interpretar retrospectivamente un registro antiguo no se introduce información futura.

### Interfaz
La pestaña muestra el número de referencias usadas, el método de referencia y la evolución cronológica. Los vectores V/S/L/C/R indican Δ normalizado o Z individual según el número de antecedentes disponibles.

### Nota práctica
En un paciente con dos registros basales, el primer registro seguirá mostrando «Sin referencia» porque no existe un registro anterior. El segundo ya utiliza el primero como referencia longitudinal.
