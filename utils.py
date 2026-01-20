def log(self, *args, **kwargs):
    """
    Función de log centralizada que verifica self.debug o self.quiet.
    Se espera que se asigne a las clases mediante self.log = log.__get__(self)
    o simplemente importándola y usándola.
    """
    # Verificamos si debemos imprimir basándonos en 'debug' o 'quiet'
    should_log = getattr(self, 'debug', False) or not getattr(self, 'quiet', True)
    
    if should_log:
        if args:
            for arg in args:
                print(arg)
        for k, v in kwargs.items():
            print(f"{k}: {v}")
