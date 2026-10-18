exportar <- function() {
args <- commandArgs(trailingOnly=TRUE)
archivo <- args[1]; carpeta <- args[2]
setwd(carpeta)
lineas <- readLines(archivo, encoding='UTF-8', warn=FALSE)
inicio <- grep('^# %% ', lineas)
final <- c(inicio[-1]-1, length(lineas))
bloques <- list()
ambiente <- globalenv()
for (i in seq_along(inicio)) {
  titulo <- sub('^# %% ', '', lineas[inicio[i]])
  codigo <- paste(lineas[seq.int(inicio[i]+1,final[i])],collapse='\n')
  cat('Celda',i,titulo,'\n'); flush.console()
  ancho <- 10; alto <- 6
  aw <- regmatches(codigo, regexpr('#\\| fig-width: *[0-9.]+',codigo))
  ah <- regmatches(codigo, regexpr('#\\| fig-height: *[0-9.]+',codigo))
  if(length(aw)&&nzchar(aw)) ancho <- as.numeric(sub('.*: *','',aw))
  if(length(ah)&&nzchar(ah)) alto <- as.numeric(sub('.*: *','',ah))
  patron <- file.path(carpeta,sprintf('fig-%02d-%%03d.svg',i))
  grDevices::svg(patron, width=ancho, height=alto,onefile=FALSE,bg='white')
  salida <- capture.output({
    for(e in parse(text=codigo)) {
      v <- withVisible(eval(e,envir=ambiente))
      if(v$visible) print(v$value)
    }
  })
  invisible(dev.off())
  outputs <- list(list(type='text',value=paste(salida,collapse='\n')))
  figs <- list.files(carpeta,pattern=sprintf('^fig-%02d-.*svg$',i),full.names=TRUE)
  for(f in figs) if(file.info(f)$size>1000) outputs[[length(outputs)+1]] <- list(type='image',value=f)
  bloques[[i]] <- list(title=titulo,prose=paste(sub("^#' ?",'',grep("^#'",strsplit(codigo,'\n')[[1]],value=TRUE)),collapse=' '),code=codigo,outputs=outputs)
}
jsonlite::write_json(bloques,file.path(carpeta,'blocks.json'),auto_unbox=TRUE,pretty=TRUE)

}
exportar()
