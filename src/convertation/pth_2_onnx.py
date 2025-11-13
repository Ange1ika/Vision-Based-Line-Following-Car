import torch
import onnx
import numpy as np
from pathlib import Path

def export_pytorch_to_onnx(
    model_path,
    output_path,
    input_size=(320, 320),
    opset_version=11,
    simplify=True,
    check=True
):
    """
    Экспорт PyTorch модели в ONNX с настройками для сегментации
    """
    print("=" * 60)
    print("ЭКСПОРТ PYTORCH -> ONNX")
    print("=" * 60)
    
    # Загрузка модели
    print(f"\n1. Загрузка модели: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Для PyTorch 2.6+ нужно добавить weights_only=False для Ultralytics моделей
    try:
        # Попытка с новым PyTorch
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Старый PyTorch
        model = torch.load(model_path, map_location=device)
    
    # Обработка разных форматов
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'ema' in model:
            model = model['ema'].module if hasattr(model['ema'], 'module') else model['ema']
    
    # Извлечение модели из DataParallel
    if hasattr(model, 'module'):
        model = model.module
    
    # Для Ultralytics моделей переключаем в режим экспорта
    if hasattr(model, 'fuse'):
        model.fuse()
    
    model.eval()
    model.to(device)
    
    print(f"   Устройство: {device}")
    print(f"   Тип модели: {type(model)}")
    
    # Создание примера входа
    print(f"\n2. Создание примера входа размером {input_size}")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Тестовый прогон
    print("\n3. Тестовый прогон модели...")
    with torch.no_grad():
        output = model(dummy_input)
        
        if isinstance(output, (list, tuple)):
            print(f"   Модель возвращает {len(output)} выходов")
            for i, out in enumerate(output):
                print(f"   Выход [{i}]: shape={out.shape}")
            output = output[0]  # Берем первый выход
        else:
            print(f"   Форма выхода: {output.shape}")
    
    # Экспорт в ONNX
    print(f"\n4. Экспорт в ONNX...")
    print(f"   Opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},  # Динамический batch
            'output': {0: 'batch'}
        } if False else None,  # Отключаем динамические оси для стабильности
        verbose=False
    )
    
    print(f"   ✓ Модель экспортирована: {output_path}")
    
    # Проверка ONNX модели
    if check:
        print("\n5. Проверка ONNX модели...")
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX модель валидна")
            
            # Информация о модели
            print("\n   Информация о графе:")
            print(f"   - Входы: {[i.name for i in onnx_model.graph.input]}")
            print(f"   - Выходы: {[o.name for o in onnx_model.graph.output]}")
            
        except Exception as e:
            print(f"   ✗ Ошибка проверки: {e}")
    
    # Упрощение модели
    if simplify:
        print("\n6. Упрощение ONNX модели...")
        try:
            import onnxsim
            
            simplified_path = str(Path(output_path).parent / f"{Path(output_path).stem}_simplified.onnx")
            
            print("   Загрузка модели для упрощения...")
            model_simp, check_ok = onnxsim.simplify(
                output_path,
                check_n=3,
                skip_fuse_bn=False,
                skip_shape_inference=False
            )
            
            if check_ok:
                onnx.save(model_simp, simplified_path)
                print(f"   ✓ Упрощенная модель сохранена: {simplified_path}")
                
                # Сравнение размеров
                original_size = Path(output_path).stat().st_size / (1024 * 1024)
                simplified_size = Path(simplified_path).stat().st_size / (1024 * 1024)
                print(f"   Размер оригинала: {original_size:.2f} MB")
                print(f"   Размер упрощенной: {simplified_size:.2f} MB")
                print(f"   Экономия: {((original_size - simplified_size) / original_size * 100):.1f}%")
            else:
                print("   ⚠️  Упрощение не прошло проверку, используйте оригинальную модель")
                
        except ImportError:
            print("   ⚠️  onnx-simplifier не установлен. Пропускаем упрощение.")
            print("   Установите: pip install onnx-simplifier")
        except Exception as e:
            print(f"   ⚠️  Ошибка упрощения: {e}")
            print("   Используйте оригинальную модель")
    
    print("\n" + "=" * 60)
    print("ЭКСПОРТ ЗАВЕРШЕН")
    print("=" * 60)
    
    return output_path

def verify_onnx_inference(onnx_path, input_size=(320, 320)):
    """
    Проверка инференса ONNX модели
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ONNX ИНФЕРЕНСА")
    print("=" * 60)
    
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Создание случайного входа
    dummy_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input})
    
    print(f"\nВход: shape={dummy_input.shape}")
    print(f"Выход: shape={output[0].shape}")
    print(f"Диапазон: [{output[0].min():.3f}, {output[0].max():.3f}]")
    
    print("\n✓ Инференс работает корректно!")
    
    return output

if __name__ == "__main__":
    # Настройки
    pytorch_model_path = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/yolo_results/best.pt"
    output_dir = Path("/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/yolo_results")
    onnx_output_path = output_dir / "best_exported.onnx"
    
    input_size = (320, 320)
    opset_version = 11  # Используйте 11 для лучшей совместимости
    
    try:
        # Экспорт
        exported_path = export_pytorch_to_onnx(
            pytorch_model_path,
            onnx_output_path,
            input_size=input_size,
            opset_version=opset_version,
            simplify=True,
            check=True
        )
        
        # Проверка инференса
        verify_onnx_inference(exported_path, input_size)
        
    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()