# Скрипт для переименования файлов с длинными именами
$trainDir = "data\images\train"
$labelsDir = "data\labels\train"

# Находим файлы с длинными именами
$longFiles = Get-ChildItem $trainDir | Where-Object {$_.Name.Length -gt 200}

foreach ($file in $longFiles) {
    $oldName = $file.FullName
    $newName = Join-Path $trainDir "car-vintage-678836.jpg"
    
    Write-Host "Переименовываем: $($file.Name)"
    Rename-Item $oldName $newName -Force
    
    # Также переименовываем соответствующий файл меток
    $labelFile = Join-Path $labelsDir ($file.BaseName + ".txt")
    if (Test-Path $labelFile) {
        $newLabelName = Join-Path $labelsDir "car-vintage-678836.txt"
        Write-Host "Переименовываем метки: $($file.BaseName).txt"
        Rename-Item $labelFile $newLabelName -Force
    }
}

Write-Host "Переименование завершено!"
