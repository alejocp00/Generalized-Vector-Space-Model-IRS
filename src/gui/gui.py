import tkinter as tk
from src.code.search_query import search_query


class GUI:
    def __init__(self) -> None:
        # Crear la ventana principal
        self.__root = tk.Tk()
        self.__root.title("Mi Buscador")

        # Crear un Frame para contener los widgets
        self.__frame = tk.Frame(self.__root)
        self.__frame.pack(padx=20, pady=20)

        # Crear una etiqueta (label)
        self.__label = tk.Label(self.__frame, text="Put your query here:")
        self.__label.pack()

        # Crear un campo de entrada (entry) para la consulta
        self.__entry = tk.Entry(self.__frame)
        self.__entry.pack()

        # Create the frame for the results
        self.__results_frame = tk.Frame(self.__root)
        self.__results_frame.pack(fill="both", expand=True)
        
        # The scroll bar and the text must be side by side
        self.__scroll_bar = tk.Scrollbar(self.__results_frame)
        self.__scroll_bar.pack(side="right", fill="y")
        
        self.__results_text = tk.Text(self.__results_frame, yscrollcommand=self.__scroll_bar.set)
        self.__results_text.pack(fill="both", expand=True)
        
        self.__scroll_bar.config(command=self.__results_text.yview)
        
        self.__results_text.config(state="disabled")
        
        # Crear un botón para realizar la búsqueda
        self.__button = tk.Button(self.__frame, text="Search", command=self.perform_search)
        self.__button.pack()
    
    # Definir una función que obtenga el texto del campo de entrada y llame a search_query
    def perform_search(self):
        query = self.__entry.get() # Obtener el texto del campo de entrada
        self.results = search_query(query) # Llamar a search_query con el texto obtenido
        self.update_results()
    
    def update_results(self):
        self.__results_text.config(state="normal")
        
        # Clear result text
        self.__results_text.delete(1.0,'end')
         
        if len(self.results) == 0:
            self.__results_text.insert('end',f"Not document found")
         
        for result in self.results:
            self.__results_text.insert('end',f"Found document: {result[0]} with a similarity of {result[1]}\n")
        
        
        self.__results_text.config(state="disabled")
    
    def run(self):
        # Ejecutar el bucle principal de la aplicación
        self.__root.mainloop()
