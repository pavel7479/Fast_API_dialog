from app.AudioProcessingPipeline import AudioProcessingPipeline

if __name__ == "__main__":

    path = "/root/audio_dialog/trancrib_txt/text_after_Gemma/temporary_2/grade_3/audio_3"

    pipeline = AudioProcessingPipeline(
    
        audio_input_folder = path,                   # папка откуда берём аудиофайлы 
        transcriber_output_folder = path + "/transcrib_2", # папка куда сохраняем расшифрованные файлы
        gemma_input_folder = path + "/transcrib_2",        # папка откуда берём файлы для анализа Gemma 
        gemma_output_folder = path + "/transcrib_2/after_gemma" # папка куда кладём файлы после анализа Gemma 
    )
    pipeline.run()
