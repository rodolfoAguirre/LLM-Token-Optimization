# LLM-Token-Optimization

## 1. Problema y alcance

Reducir el consumo de tokens de un asistente de soporte técnico seleccionando la información necesaria para responder preguntas sobre documentos.

Las entradas serán una pregunta, documentos divididos en fragmentos y un presupuesto de tokens. La salida será el contexto seleccionado para el LLM. Inicialmente, las dependencias entre fragmentos se verificarán manualmente.

## 2. Algoritmo propuesto

Se utilizará un algoritmo voraz que seleccione fragmentos y sus dependencias según su utilidad adicional por token, considerando relevancia y redundancia.

El algoritmo respetará el presupuesto, evitará duplicados y conservará las dependencias declaradas. Tendrá reglas de desempate deterministas, aunque no garantizará la solución óptima ni la calidad de la respuesta.

## 3. Complejidad y recursos

La implementación prevista tendrá complejidad temporal O(n³) y espacial O(n²), donde n es el número de fragmentos. Esto supone dependencias y similitudes precalculadas y utilidades actualizadas en memoria.

El procesamiento inicial y la ejecución del LLM se medirán por separado. La versión inicial funcionará secuencialmente en CPU, sin HPC.

## 4. Evaluación

Se comparará la propuesta con dos alternativas: enviar toda la documentación cuando quepa en el modelo y seleccionar fragmentos únicamente por relevancia.

Se aumentará la cantidad de documentos y consultas y se reducirán los presupuestos. Se medirán tokens, tiempo, memoria, costo, exactitud, completitud y consistencia.

Se utilizarán preguntas y respuestas de referencia fijas, la misma configuración del LLM y cinco ejecuciones por consulta. Se guardarán configuraciones, versiones y resultados para reproducir los experimentos.

## 5. Limitaciones y mejoras

Las principales limitaciones son la falta de optimalidad, el costo al aumentar los fragmentos y la dependencia de estimaciones de relevancia y relaciones manuales.

Se evaluarán tres mejoras: filtrar candidatos, intercambiar fragmentos para mejorar la selección y detectar dependencias automáticamente. Cada mejora se probará por separado para medir su efecto sobre los recursos y la calidad.
