"""
Algoritmo de Dijkstra - Implementación en Python
=================================================

Descripción general
-------------------
El algoritmo de Dijkstra resuelve el problema del camino más corto
desde un único origen (Single-Source Shortest Path, SSSP) en un grafo
dirigido o no dirigido con pesos **no negativos** en sus aristas.

Fue publicado por Edsger W. Dijkstra en 1959 en el artículo
"A note on two problems in connexion with graphs".

Funcionamiento paso a paso
---------------------------
1. Inicializar dist[origen] = 0 y dist[v] = ∞ para todo v ≠ origen.
2. Insertar todos los nodos en una cola de prioridad (min-heap) ordenada
   por distancia tentativa.
3. Mientras la cola no esté vacía:
   a. Extraer el nodo u con menor distancia tentativa.
   b. Marcarlo como "visitado" (ya no puede mejorar).
   c. Para cada vecino v de u:
      - Calcular dist_alt = dist[u] + peso(u, v).
      - Si dist_alt < dist[v], actualizar dist[v] y registrar previo[v] = u.
      - Reinsertar v en el heap con la nueva prioridad.
4. Al finalizar, dist[v] contiene la distancia mínima desde origen hasta v
   y previo[v] permite reconstruir la ruta.

Complejidad
-----------
+--------------------+---------------------+
| Implementación     | Complejidad         |
+--------------------+---------------------+
| Lista simple       | O(V²)               |
| Min-heap (heapq)   | O((V + E) log V)    |
| Heap de Fibonacci  | O(E + V log V)      |
+--------------------+---------------------+

Esta implementación usa ``heapq`` de Python (min-heap binario).
Es la opción más práctica para grafos dispersos (E << V²).

Limitaciones
------------
- NO funciona con aristas de peso negativo.
  Para ese caso usar el algoritmo de Bellman-Ford.
- NO detecta ciclos negativos.
- Asume que todos los pesos son enteros o flotantes ≥ 0.

Aplicaciones reales
-------------------
- GPS y navegación (Google Maps, OpenStreetMap).
- Enrutamiento de paquetes en redes (OSPF, IS-IS).
- Planificación de rutas en robótica.
- Juegos de video (pathfinding de NPCs).
- Análisis de redes sociales (camino más corto entre usuarios).

Uso rápido
----------
    from dijkstra import Grafo

    g = Grafo()
    g.agregar_arista("A", "B", 4)
    g.agregar_arista("A", "C", 2)
    g.agregar_arista("B", "C", 1)

    g.camino_minimo("A", "B")
    # ✓ Camino óptimo 'A' → 'B': A → C → B  (costo=3)

Dependencias
------------
Solo módulos de la biblioteca estándar de Python:
    - heapq       : cola de prioridad (min-heap binario)
    - collections : defaultdict para la lista de adyacencia
    - typing      : anotaciones de tipo (Optional)

Compatibilidad
--------------
Python 3.9+  (usa sintaxis de type hints nativos: dict[...], list[...])

Autor
-----
Generado como material académico para el curso de Estructuras de Datos.
Referencia original: Dijkstra, E. W. (1959). A note on two problems in
connexion with graphs. Numerische Mathematik, 1(1), 269-271.
"""

import heapq
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class Grafo:
    """
    Grafo no dirigido ponderado representado mediante lista de adyacencia.

    Internamente cada nodo mapea a una lista de tuplas ``(vecino, peso)``.
    Esta estructura es eficiente en memoria para grafos dispersos y permite
    iterar sobre los vecinos de un nodo en O(grado(nodo)).

    Atributos
    ----------
    adyacencia : dict[str, list[tuple[str, int]]]
        Mapeo nodo → lista de (vecino, peso).
    nodos : set[str]
        Conjunto de todos los nodos registrados en el grafo.

    Ejemplo
    -------
    >>> g = Grafo()
    >>> g.agregar_arista("X", "Y", 5)
    >>> g.agregar_arista("Y", "Z", 3)
    >>> g.nodos
    {'X', 'Y', 'Z'}
    >>> g.adyacencia["Y"]
    [('X', 5), ('Z', 3)]
    """

    def __init__(self) -> None:
        """Inicializa un grafo vacío."""
        self.adyacencia: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.nodos: set[str] = set()

    # ------------------------------------------------------------------
    # Construcción del grafo
    # ------------------------------------------------------------------

    def agregar_arista(self, u: str, v: str, peso: int) -> None:
        """
        Agrega una arista **bidireccional** (no dirigida) entre ``u`` y ``v``
        con el peso especificado.

        Dado que el grafo es no dirigido, se registran dos entradas en la
        lista de adyacencia: u → v y v → u, ambas con el mismo peso.

        Parámetros
        ----------
        u : str
            Nodo origen de la arista.
        v : str
            Nodo destino de la arista.
        peso : int
            Peso o costo de recorrer la arista.  Debe ser ≥ 0.

        Raises
        ------
        ValueError
            Si ``peso`` es negativo (no soportado por Dijkstra).

        Ejemplo
        -------
        >>> g = Grafo()
        >>> g.agregar_arista("A", "B", 7)
        >>> ("B", 7) in g.adyacencia["A"]
        True
        >>> ("A", 7) in g.adyacencia["B"]
        True
        """
        if peso < 0:
            raise ValueError(
                f"Peso negativo no permitido en Dijkstra: arista ({u}, {v}, {peso}). "
                "Usa Bellman-Ford para grafos con pesos negativos."
            )
        self.adyacencia[u].append((v, peso))
        self.adyacencia[v].append((u, peso))
        self.nodos.update([u, v])

    # ------------------------------------------------------------------
    # Algoritmo de Dijkstra
    # ------------------------------------------------------------------

    def dijkstra(
        self, origen: str
    ) -> tuple[dict[str, int], dict[str, Optional[str]]]:
        """
        Ejecuta el algoritmo de Dijkstra desde el nodo ``origen`` y calcula
        la distancia mínima hacia **todos** los nodos alcanzables del grafo.

        Implementación con min-heap (``heapq``) para garantizar
        complejidad O((V + E) log V).

        Estrategia de relajación
        ~~~~~~~~~~~~~~~~~~~~~~~~
        Para cada nodo ``u`` extraído del heap y cada vecino ``v``:

            dist_alt = dist[u] + peso(u, v)
            si dist_alt < dist[v]:
                dist[v] = dist_alt   ← relajación exitosa
                previo[v] = u        ← registrar predecesor

        El heap puede contener entradas duplicadas para un mismo nodo
        (con distancias distintas). Se descartan las entradas obsoletas
        verificando si el nodo ya fue visitado.

        Parámetros
        ----------
        origen : str
            Nodo de partida del algoritmo.

        Retorna
        -------
        distancias : dict[str, int]
            Distancia mínima desde ``origen`` a cada nodo.
            El valor es ``float('inf')`` si el nodo no es alcanzable.
        previo : dict[str, Optional[str]]
            Para cada nodo, el nodo anterior en el camino óptimo.
            ``None`` si es el origen o si el nodo no es alcanzable.

        Raises
        ------
        KeyError
            Si ``origen`` no existe en el grafo.

        Notas
        -----
        - Los nodos aislados (sin aristas) aparecen en ``distancias``
          con valor ``inf``.
        - El heap puede crecer hasta O(E) entradas en el peor caso.

        Ejemplo
        -------
        >>> g = Grafo()
        >>> g.agregar_arista("A", "B", 1)
        >>> g.agregar_arista("B", "C", 2)
        >>> dist, prev = g.dijkstra("A")
        >>> dist["C"]
        3
        >>> prev["C"]
        'B'
        """
        if origen not in self.nodos:
            raise KeyError(f"El nodo origen '{origen}' no existe en el grafo.")

        # Inicializar distancias: 0 para el origen, ∞ para el resto
        distancias: dict[str, int] = {n: float("inf") for n in self.nodos}
        distancias[origen] = 0

        # previo[v] = nodo anterior a v en el camino óptimo
        previo: dict[str, Optional[str]] = {n: None for n in self.nodos}

        # Min-heap de tuplas (distancia_acumulada, nodo)
        # heapq en Python implementa un min-heap binario
        heap: list[tuple[int, str]] = [(0, origen)]

        # Conjunto de nodos ya procesados (extraídos con distancia definitiva)
        visitados: set[str] = set()

        # ── Cabecera de la consola paso a paso ──────────────────────────
        print(f"\n{'='*60}")
        print(f"  Dijkstra — origen: '{origen}'")
        print(f"{'='*60}")
        print(f"  {'Paso':<5} {'Acción':<40} {'Heap size'}")
        print(f"  {'-'*55}")

        paso = 1

        while heap:
            # Extraer el nodo con menor distancia tentativa (operación O(log n))
            dist_u, u = heapq.heappop(heap)

            # Entrada obsoleta: el nodo ya fue procesado con una distancia menor
            if u in visitados:
                continue

            visitados.add(u)
            print(
                f"  {paso:<5} ✔ Visitar '{u}' (dist_mín={dist_u})"
                f"{'':>10} heap={len(heap)}"
            )

            # Relajar aristas hacia los vecinos no visitados
            for v, peso in self.adyacencia[u]:
                if v in visitados:
                    continue

                dist_alt = dist_u + peso

                if dist_alt < distancias[v]:
                    anterior = distancias[v]
                    distancias[v] = dist_alt
                    previo[v] = u

                    # Insertar en el heap con la nueva prioridad (O(log n))
                    heapq.heappush(heap, (dist_alt, v))

                    ant_str = "∞" if anterior == float("inf") else anterior
                    print(
                        f"  {'':5}   → Relajar '{u}'→'{v}': "
                        f"{ant_str} → {dist_alt}  ({dist_u}+{peso})"
                    )

            paso += 1

        print(f"  {'-'*55}")
        print(f"  Nodos visitados: {sorted(visitados)}")
        print()

        return distancias, previo

    # ------------------------------------------------------------------
    # Reconstrucción de rutas
    # ------------------------------------------------------------------

    def camino(
        self, previo: dict[str, Optional[str]], destino: str
    ) -> list[str]:
        """
        Reconstruye el camino mínimo desde el origen hasta ``destino``
        siguiendo los punteros del diccionario ``previo``.

        El recorrido parte desde ``destino`` y sigue los predecesores
        hacia atrás hasta llegar al nodo cuyo predecesor es ``None``
        (el origen). La lista resultante se invierte al final.

        Parámetros
        ----------
        previo : dict[str, Optional[str]]
            Diccionario de predecesores generado por :meth:`dijkstra`.
        destino : str
            Nodo al que se desea llegar.

        Retorna
        -------
        list[str]
            Lista ordenada de nodos que forman el camino mínimo,
            incluyendo origen y destino. Lista vacía si ``destino``
            no es alcanzable (predecesor nunca fue actualizado pero
            el nodo de inicio no coincide con el destino).

        Ejemplo
        -------
        >>> g = Grafo()
        >>> g.agregar_arista("A", "B", 1)
        >>> g.agregar_arista("B", "C", 2)
        >>> _, prev = g.dijkstra("A")
        >>> g.camino(prev, "C")
        ['A', 'B', 'C']
        """
        ruta: list[str] = []
        nodo: Optional[str] = destino

        # Seguir los punteros de previo hasta llegar al origen (None)
        while nodo is not None:
            ruta.append(nodo)
            nodo = previo[nodo]

        ruta.reverse()

        # Si la ruta no arranca en un nodo sin predecesor válido,
        # significa que destino no era alcanzable
        return ruta

    # ------------------------------------------------------------------
    # Método combinado (dijkstra + reconstrucción + reporte)
    # ------------------------------------------------------------------

    def camino_minimo(self, origen: str, destino: str) -> None:
        """
        Método de conveniencia que ejecuta Dijkstra e imprime en consola
        las distancias a todos los nodos y el camino óptimo hacia ``destino``.

        Combina :meth:`dijkstra` y :meth:`camino` en una sola llamada.

        Parámetros
        ----------
        origen : str
            Nodo de partida.
        destino : str
            Nodo al que se desea llegar.

        Retorna
        -------
        None
            Solo imprime resultados en consola.  Para obtener los valores
            de forma programática usa :meth:`dijkstra` directamente.

        Ejemplo
        -------
        >>> g = Grafo()
        >>> g.agregar_arista("A", "B", 4)
        >>> g.agregar_arista("A", "C", 2)
        >>> g.agregar_arista("B", "C", 1)
        >>> g.camino_minimo("A", "B")
        # Imprime paso a paso y resultado: A → C → B  (costo=3)
        """
        distancias, previo = self.dijkstra(origen)

        # Tabla de distancias finales
        print(f"  Distancias mínimas desde '{origen}':")
        for nodo, d in sorted(distancias.items()):
            alcanzable = d != float("inf")
            simbolo = "✓" if alcanzable else "✗"
            valor = str(d) if alcanzable else "∞  (no alcanzable)"
            print(f"    {simbolo} {nodo}: {valor}")

        print()

        # Resultado del camino origen → destino
        if distancias[destino] == float("inf"):
            print(f"  ✗ No existe camino de '{origen}' a '{destino}'.")
        else:
            ruta = self.camino(previo, destino)
            print(f"  ✓ Camino óptimo '{origen}' → '{destino}':")
            print(f"    Ruta  : {' → '.join(ruta)}")
            print(f"    Costo : {distancias[destino]}")

        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Funciones de ejemplo
# ---------------------------------------------------------------------------

def ejemplo_ciudad() -> None:
    """
    Grafo tipo «mapa de ciudad» con 9 nodos (A–I) y 15 aristas.

    Topología::

        A -4- B -5- D
        |     |     |
        2     1     2
        |     |     |
        C -8- D -6- F -1- G
              |     |     |
             10     3     2
              |     |     |
              E-----+     H -1- I
              |           |
              7-----------G

    Se busca el camino mínimo de A a I.
    Resultado esperado: A → C → B → D → E → F → G → H → I  (costo = 13)
    """
    g = Grafo()
    aristas = [
        ("A", "B", 4),  ("A", "C", 2),
        ("B", "C", 1),  ("B", "D", 5),
        ("C", "D", 8),  ("C", "E", 10),
        ("D", "E", 2),  ("D", "F", 6),
        ("E", "F", 3),  ("E", "G", 7),
        ("F", "G", 1),  ("F", "H", 4),
        ("G", "H", 2),  ("G", "I", 5),
        ("H", "I", 1),
    ]
    for u, v, w in aristas:
        g.agregar_arista(u, v, w)

    g.camino_minimo("A", "I")


def ejemplo_simple() -> None:
    """
    Grafo clásico de 6 nodos basado en el ejemplo original de Dijkstra (1959).

    Nodos: A B C D E F
    Aristas y pesos::

        A -7-  B
        A -9-  C
        A -14- F
        B -10- C
        B -15- D
        C -11- D
        C -2-  F
        D -6-  E
        E -9-  F

    Se busca el camino mínimo de A a E.
    Resultado esperado: A → C → D → E  (costo = 26)
    """
    g = Grafo()
    aristas = [
        ("A", "B", 7),  ("A", "C", 9),  ("A", "F", 14),
        ("B", "C", 10), ("B", "D", 15),
        ("C", "D", 11), ("C", "F", 2),
        ("D", "E", 6),
        ("E", "F", 9),
    ]
    for u, v, w in aristas:
        g.agregar_arista(u, v, w)

    g.camino_minimo("A", "E")


def ejemplo_complejo() -> None:
    """
    Grafo complejo con 11 nodos (A–K) y 16 aristas para pruebas más exigentes.

    Se busca el camino mínimo de A a K.
    Resultado esperado: A → B → E → G → H → K  (costo = 20)  *(o equivalente)*
    """
    g = Grafo()
    aristas = [
        ("A", "B", 2),  ("A", "C", 6),
        ("B", "D", 5),  ("B", "E", 8),
        ("C", "D", 3),  ("C", "F", 7),
        ("D", "G", 4),
        ("E", "G", 2),  ("E", "H", 9),
        ("F", "G", 1),  ("F", "I", 5),
        ("G", "H", 3),  ("G", "J", 6),
        ("H", "K", 4),
        ("I", "J", 2),
        ("J", "K", 3),
    ]
    for u, v, w in aristas:
        g.agregar_arista(u, v, w)

    g.camino_minimo("A", "K")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  SIMULADOR DE DIJKSTRA — Estructuras de Datos")
    print("█" * 60)

    print("\n>>> Ejemplo 1: Grafo Ciudad (9 nodos, 15 aristas)")
    ejemplo_ciudad()

    print(">>> Ejemplo 2: Grafo Clásico Dijkstra 1959 (6 nodos, 9 aristas)")
    ejemplo_simple()

    print(">>> Ejemplo 3: Grafo Complejo (11 nodos, 16 aristas)")
    ejemplo_complejo()