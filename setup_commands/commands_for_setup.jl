using PkgTemplates;
t = Template(;
    user="radekszewczyk01",
    license="MIT",
    authors=["Radek"],
    plugins=[
        TravisCI(),
        Codecov(),
        Coveralls(),
        AppVeyor()
    ]
)

generate("myExample", t)