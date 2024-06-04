use report_ai_cctv;

CREATE TABLE activity_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(8),
    employee_name VARCHAR(50),
    wrapping_time VARCHAR(8),
    unloading_time VARCHAR(8),
    packing_time VARCHAR(8),
    sorting_time VARCHAR(8),
    absent_person VARCHAR(255)
);

SELECT * FROM activity_log;

delete from activity_log where id>0;

ALTER TABLE activity_log AUTO_INCREMENT = 1;